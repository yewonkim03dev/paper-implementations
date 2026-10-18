from torch import Tensor, nn

from .decoder import Decoder
from .encoder import Encoder
from .masks import create_causal_mask, create_padding_mask


class Transformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        num_heads: int,
        d_ff: int,
        num_layers: int,
        dropout: float = 0.1,
        max_len: int = 5000,
        padding_index: int = 0,
    ):
        super().__init__()

        self.padding_index = padding_index
        self.config = {
            "vocab_size": vocab_size,
            "d_model": d_model,
            "num_heads": num_heads,
            "d_ff": d_ff,
            "num_layers": num_layers,
            "dropout": dropout,
            "max_len": max_len,
            "padding_index": padding_index,
        }

        self.encoder = Encoder(
            vocab_size,
            d_model,
            num_heads,
            d_ff,
            num_layers,
            dropout,
            max_len,
        )
        self.decoder = Decoder(
            vocab_size,
            d_model,
            num_heads,
            d_ff,
            num_layers,
            dropout,
            max_len,
        )
        self.output_projection = nn.Linear(
            d_model,
            vocab_size,
            bias=False,
        )

        # 원논문처럼 source/target embedding과 출력 projection의 weight를 공유한다.
        shared_embedding_weight = self.encoder.token_embedding.embedding.weight
        self.decoder.token_embedding.embedding.weight = shared_embedding_weight
        self.output_projection.weight = shared_embedding_weight

    def forward(
        self,
        source_tokens: Tensor,
        target_tokens: Tensor,
    ) -> tuple[
        Tensor,
        list[Tensor],
        list[Tensor],
        list[Tensor],
    ]:
        # source_tokens: (B, S_source)
        # target_tokens: (B, S_target)

        # source의 padding key를 Encoder와 Cross-Attention에서 가린다.
        # (B, S_source) -> (B, 1, 1, S_source)
        source_mask = create_padding_mask(
            source_tokens,
            self.padding_index,
        )

        # target의 padding key와 미래 token을 함께 가린다.
        # padding mask: (B, 1, 1, S_target)
        target_padding_mask = create_padding_mask(
            target_tokens,
            self.padding_index,
        )
        # causal mask: (1, 1, S_target, S_target)
        target_causal_mask = create_causal_mask(
            target_tokens.size(1),
            target_tokens.device,
        )
        # broadcast 후: (B, 1, S_target, S_target)
        target_mask = target_padding_mask | target_causal_mask

        # Encoder: (B, S_source) -> (B, S_source, D_model)
        encoder_output, encoder_attention_weights = self.encoder(
            source_tokens,
            source_mask,
        )

        # Decoder: (B, S_target) -> (B, S_target, D_model)
        decoder_output, decoder_self_weights, decoder_cross_weights = (
            self.decoder(
                target_tokens,
                encoder_output,
                target_mask,
                source_mask,
            )
        )

        # Linear projection: (B, S_target, D_model)
        # -> (B, S_target, vocab_size)
        logits = self.output_projection(decoder_output)

        # softmax는 학습 시 CrossEntropyLoss 내부에서 처리한다.
        return (
            logits,
            encoder_attention_weights,
            decoder_self_weights,
            decoder_cross_weights,
        )
