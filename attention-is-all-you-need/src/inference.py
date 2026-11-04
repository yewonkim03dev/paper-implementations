from pathlib import Path

import torch
from torch import Tensor
from torch.nn import functional as F

from .masks import create_causal_mask, create_padding_mask
from .tokenizer import BOS_INDEX, BPETokenizer, EOS_INDEX
from .transformer import Transformer


def _encode_source(
    model: Transformer,
    source_tokens: Tensor,
) -> tuple[Tensor, Tensor]:
    # source_tokens: (B, S_source)
    source_mask = create_padding_mask(
        source_tokens,
        model.padding_index,
    )
    encoder_output, _ = model.encoder(source_tokens, source_mask)

    # return (B, S_source, D_model), (B, 1, 1, S_source)
    return encoder_output, source_mask


def _next_token_logits(
    model: Transformer,
    generated_tokens: Tensor,
    encoder_output: Tensor,
    source_mask: Tensor,
) -> Tensor:
    # generated_tokens: (B, S_generated)
    target_padding_mask = create_padding_mask(
        generated_tokens,
        model.padding_index,
    )
    target_causal_mask = create_causal_mask(
        generated_tokens.size(1),
        generated_tokens.device,
    )
    target_mask = target_padding_mask | target_causal_mask

    decoder_output, _, _ = model.decoder(
        generated_tokens,
        encoder_output,
        target_mask,
        source_mask,
    )

    # 마지막 위치: (B, D_model) -> (B, vocab_size)
    return model.output_projection(decoder_output[:, -1])


@torch.no_grad()
def greedy_decode(
    model: Transformer,
    source_tokens: Tensor,
    bos_index: int = BOS_INDEX,
    eos_index: int = EOS_INDEX,
    max_length: int = 256,
) -> Tensor:
    if source_tokens.dim() != 2:
        raise ValueError("source_tokens must have shape (B, S_source)")
    if max_length < 2:
        raise ValueError("max_length must be at least 2")

    model.eval()
    device = next(model.parameters()).device
    source_tokens = source_tokens.to(device)
    batch_size = source_tokens.size(0)

    encoder_output, source_mask = _encode_source(model, source_tokens)
    generated_tokens = torch.full(
        (batch_size, 1),
        bos_index,
        dtype=torch.long,
        device=device,
    )
    finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

    for _ in range(max_length - 1):
        # (B, vocab_size)
        next_token_logits = _next_token_logits(
            model,
            generated_tokens,
            encoder_output,
            source_mask,
        )

        # 첫 token 이후에는 <pad>와 <bos>를 생성 후보에서 제외한다.
        next_token_logits[:, model.padding_index] = float("-inf")
        next_token_logits[:, bos_index] = float("-inf")
        next_tokens = next_token_logits.argmax(dim=-1)

        # 이미 <eos>를 생성한 문장은 남은 위치를 <pad>로 채운다.
        next_tokens = torch.where(
            finished,
            torch.full_like(next_tokens, model.padding_index),
            next_tokens,
        )
        generated_tokens = torch.cat(
            [generated_tokens, next_tokens.unsqueeze(1)],
            dim=1,
        )

        finished = finished | (next_tokens == eos_index)
        if bool(finished.all()):
            break

    # (B, S_generated)
    return generated_tokens


def _length_normalized_score(
    log_probability: float,
    generated_length: int,
    length_penalty: float,
) -> float:
    penalty = ((5.0 + generated_length) / 6.0) ** length_penalty
    return log_probability / penalty


@torch.no_grad()
def beam_search_decode(
    model: Transformer,
    source_tokens: Tensor,
    bos_index: int = BOS_INDEX,
    eos_index: int = EOS_INDEX,
    max_length: int = 256,
    beam_size: int = 4,
    length_penalty: float = 0.6,
) -> Tensor:
    if source_tokens.dim() != 2 or source_tokens.size(0) != 1:
        raise ValueError("beam search expects source_tokens with shape (1, S_source)")
    if max_length < 2:
        raise ValueError("max_length must be at least 2")
    if beam_size <= 0:
        raise ValueError("beam_size must be greater than 0")
    if length_penalty < 0:
        raise ValueError("length_penalty must not be negative")

    model.eval()
    device = next(model.parameters()).device
    source_tokens = source_tokens.to(device)
    encoder_output, source_mask = _encode_source(model, source_tokens)

    # (generated token IDs, 누적 log probability, 종료 여부)
    beams: list[tuple[Tensor, float, bool]] = [
        (
            torch.tensor([bos_index], dtype=torch.long, device=device),
            0.0,
            False,
        )
    ]

    for _ in range(max_length - 1):
        candidates: list[tuple[Tensor, float, bool]] = []

        for tokens, score, finished in beams:
            if finished:
                candidates.append((tokens, score, True))
                continue

            next_token_logits = _next_token_logits(
                model,
                tokens.unsqueeze(0),
                encoder_output,
                source_mask,
            )[0]
            next_token_logits[model.padding_index] = float("-inf")
            next_token_logits[bos_index] = float("-inf")

            log_probabilities = F.log_softmax(next_token_logits, dim=-1)
            top_log_probabilities, top_indices = torch.topk(
                log_probabilities,
                k=min(beam_size, log_probabilities.size(0)),
            )

            for log_probability, token_index in zip(
                top_log_probabilities.tolist(),
                top_indices.tolist(),
            ):
                next_tokens = torch.cat(
                    [
                        tokens,
                        torch.tensor(
                            [token_index],
                            dtype=torch.long,
                            device=device,
                        ),
                    ]
                )
                candidates.append(
                    (
                        next_tokens,
                        score + log_probability,
                        token_index == eos_index,
                    )
                )

        beams = sorted(
            candidates,
            key=lambda beam: _length_normalized_score(
                beam[1],
                max(1, beam[0].numel() - 1),
                length_penalty,
            ),
            reverse=True,
        )[:beam_size]

        if all(finished for _, _, finished in beams):
            break

    best_tokens, _, _ = max(
        beams,
        key=lambda beam: _length_normalized_score(
            beam[1],
            max(1, beam[0].numel() - 1),
            length_penalty,
        ),
    )
    return best_tokens.unsqueeze(0)


def translate_text(
    model: Transformer,
    tokenizer: BPETokenizer,
    text: str,
    max_length: int = 256,
    beam_size: int = 4,
    length_penalty: float = 0.6,
) -> str:
    device = next(model.parameters()).device
    source_ids = tokenizer.encode_source(text)
    source_tokens = torch.tensor(
        [source_ids],
        dtype=torch.long,
        device=device,
    )

    if beam_size == 1:
        generated_tokens = greedy_decode(
            model,
            source_tokens,
            bos_index=tokenizer.bos_index,
            eos_index=tokenizer.eos_index,
            max_length=max_length,
        )
    else:
        generated_tokens = beam_search_decode(
            model,
            source_tokens,
            bos_index=tokenizer.bos_index,
            eos_index=tokenizer.eos_index,
            max_length=max_length,
            beam_size=beam_size,
            length_penalty=length_penalty,
        )

    return tokenizer.decode(generated_tokens[0].cpu())


def load_transformer_checkpoint(
    checkpoint_path: str | Path,
    device: torch.device | str = "cpu",
) -> Transformer:
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True,
    )
    model_config = checkpoint.get("model_config")
    if not isinstance(model_config, dict):
        raise ValueError("checkpoint does not contain model_config")

    model = Transformer(**model_config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model
