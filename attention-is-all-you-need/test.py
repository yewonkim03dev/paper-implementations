"""Local validation entry point for the Attention Is All You Need implementation."""

from pathlib import Path
import tempfile

import torch
from torch.utils.data import DataLoader

from src.attention import MultiHeadAttention, ScaledDotProductAttention
from src.data import TranslationDataset, collate_translation_batch
from src.decoder import Decoder, DecoderLayer
from src.embedding import PositionalEncoding, TokenEmbedding
from src.encoder import Encoder, EncoderLayer
from src.feed_forward import PositionwiseFeedForward
from src.masks import create_causal_mask, create_padding_mask
from src.tokenizer import (
    BOS_INDEX,
    BPETokenizer,
    EOS_INDEX,
    PAD_INDEX,
    UNK_INDEX,
    load_parallel_corpus,
    normalize_text,
    preprocess_pairs,
)
from src.training import (
    LabelSmoothingLoss,
    create_transformer_optimizer,
    evaluate,
    save_training_checkpoint,
    train_epoch,
)
from src.transformer import Transformer


def test_train_epoch() -> None:
    vocab_size = 16
    model = Transformer(
        vocab_size=vocab_size,
        d_model=16,
        num_heads=4,
        d_ff=32,
        num_layers=1,
        dropout=0.0,
        padding_index=PAD_INDEX,
    )
    loss_function = LabelSmoothingLoss(
        smoothing=0.1,
        padding_index=PAD_INDEX,
    )
    optimizer, scheduler = create_transformer_optimizer(
        model.parameters(),
        d_model=16,
        warmup_steps=4,
    )

    batch = (
        torch.tensor(
            [
                [4, 5, EOS_INDEX, PAD_INDEX],
                [6, 7, 8, EOS_INDEX],
            ]
        ),
        torch.tensor(
            [
                [BOS_INDEX, 9, PAD_INDEX],
                [BOS_INDEX, 10, 11],
            ]
        ),
        torch.tensor(
            [
                [9, EOS_INDEX, PAD_INDEX],
                [10, 11, EOS_INDEX],
            ]
        ),
    )
    parameter = model.encoder.layers[0].self_attention.query_projection.weight
    parameter_before_training = parameter.detach().clone()

    average_loss = train_epoch(
        model,
        [batch, batch],
        loss_function,
        optimizer,
        scheduler,
        device="cpu",
        log_interval=0,
    )

    assert average_loss > 0
    assert scheduler.step_number == 2
    assert not torch.equal(parameter, parameter_before_training)

    # validation에서는 parameter와 gradient가 변경되지 않아야 함
    optimizer.zero_grad(set_to_none=True)
    parameter_before_validation = parameter.detach().clone()
    validation_loss = evaluate(
        model,
        [batch],
        loss_function,
        device="cpu",
    )
    assert validation_loss > 0
    assert not model.training
    assert torch.equal(parameter, parameter_before_validation)
    assert all(
        model_parameter.grad is None
        for model_parameter in model.parameters()
    )

    with tempfile.TemporaryDirectory() as tmp_dir:
        checkpoint_path = Path(tmp_dir) / "epoch_001.pt"
        save_training_checkpoint(
            checkpoint_path,
            model,
            optimizer,
            scheduler,
            epoch=1,
            loss=average_loss,
        )

        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=True,
        )
        assert checkpoint["epoch"] == 1
        assert checkpoint["loss"] == average_loss
        assert checkpoint["scheduler_state_dict"]["step_number"] == 2
        assert "model_state_dict" in checkpoint
        assert "optimizer_state_dict" in checkpoint

    print("Train epoch test passed")


def test_training_setup() -> None:
    B = 2
    S_target = 4
    vocab_size = 12

    loss_function = LabelSmoothingLoss(
        smoothing=0.1,
        padding_index=PAD_INDEX,
    )
    logits = torch.randn(
        B,
        S_target,
        vocab_size,
        requires_grad=True,
    )
    labels = torch.tensor(
        [
            [4, 5, EOS_INDEX, PAD_INDEX],
            [6, 7, 8, EOS_INDEX],
        ]
    )

    loss = loss_function(logits, labels)
    assert loss.dim() == 0
    assert torch.isfinite(loss)

    # padding 위치의 logit은 loss에 영향을 주지 않아야 함
    changed_logits = logits.detach().clone()
    changed_logits[labels == PAD_INDEX] = torch.randn(vocab_size) * 100
    changed_loss = loss_function(changed_logits, labels)
    assert torch.allclose(loss.detach(), changed_loss)

    loss.backward()
    assert logits.grad is not None
    assert torch.all(logits.grad[labels == PAD_INDEX] == 0)
    assert torch.any(logits.grad[labels != PAD_INDEX] != 0)

    model = Transformer(
        vocab_size=vocab_size,
        d_model=16,
        num_heads=4,
        d_ff=32,
        num_layers=1,
        dropout=0.0,
        padding_index=PAD_INDEX,
    )
    optimizer, scheduler = create_transformer_optimizer(
        model.parameters(),
        d_model=16,
        warmup_steps=4,
    )

    assert optimizer.defaults["betas"] == (0.9, 0.98)
    assert optimizer.defaults["eps"] == 1e-9
    assert optimizer.param_groups[0]["lr"] == 0.0

    # warmup 동안 증가하고 이후 step의 제곱근에 반비례해 감소해야 함
    assert scheduler.rate(1) < scheduler.rate(2)
    assert scheduler.rate(3) < scheduler.rate(4)
    assert scheduler.rate(5) < scheduler.rate(4)

    source_tokens = torch.tensor([[4, 5, EOS_INDEX, PAD_INDEX]])
    decoder_input = torch.tensor([[BOS_INDEX, 6, 7]])
    target_labels = torch.tensor([[6, 7, EOS_INDEX]])

    optimizer.zero_grad()
    model_logits, _, _, _ = model(source_tokens, decoder_input)
    model_loss = loss_function(model_logits, target_labels)
    model_loss.backward()

    parameter = model.encoder.layers[0].self_attention.query_projection.weight
    parameter_before_step = parameter.detach().clone()

    # learning rate를 먼저 설정한 뒤 optimizer가 parameter를 갱신한다.
    learning_rate = scheduler.step()
    optimizer.step()

    assert learning_rate == scheduler.rate(1)
    assert optimizer.param_groups[0]["lr"] == learning_rate
    assert not torch.equal(parameter, parameter_before_step)

    print("Training setup test passed")


def test_translation_dataset() -> None:
    pairs = [
        ("A small cat.", "Eine kleine Katze."),
        ("Two dogs run across the field.", "Zwei Hunde laufen über das Feld."),
        ("A child smiles.", "Ein Kind lächelt."),
    ]

    tokenizer = BPETokenizer()
    tokenizer.train_from_pairs(
        pairs,
        vocab_size=100,
        min_frequency=1,
    )

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        src_file = tmp_path / "train.en"
        tgt_file = tmp_path / "train.de"

        src_file.write_text(
            "\n".join(source for source, _ in pairs),
            encoding="utf-8",
        )
        tgt_file.write_text(
            "\n".join(target for _, target in pairs),
            encoding="utf-8",
        )

        dataset = TranslationDataset(
            src_file,
            tgt_file,
            tokenizer,
        )
        samples = list(dataset)

        assert len(samples) == len(pairs)
        for source, decoder_input, labels in samples:
            assert source.dtype == torch.long
            assert decoder_input.dtype == torch.long
            assert labels.dtype == torch.long

            # target을 한 칸 이동해 Decoder 입력과 정답을 만들어야 함
            assert decoder_input[0].item() == BOS_INDEX
            assert labels[-1].item() == EOS_INDEX
            assert torch.equal(decoder_input[1:], labels[:-1])

        data_loader = DataLoader(
            dataset,
            batch_size=2,
            collate_fn=collate_translation_batch,
        )
        source_batch, decoder_input_batch, label_batch = next(
            iter(data_loader)
        )

        # batch별 최대 길이에 맞춰 padding된 shape 확인
        assert source_batch.dim() == 2
        assert decoder_input_batch.dim() == 2
        assert label_batch.shape == decoder_input_batch.shape
        assert source_batch.size(0) == 2
        assert decoder_input_batch.size(0) == 2

        for row, (source, _, _) in zip(source_batch, samples[:2]):
            assert torch.equal(row[: source.size(0)], source)
            assert torch.all(row[source.size(0) :] == PAD_INDEX)

        for row, (_, decoder_input, _) in zip(
            decoder_input_batch,
            samples[:2],
        ):
            assert torch.equal(row[: decoder_input.size(0)], decoder_input)
            assert torch.all(row[decoder_input.size(0) :] == PAD_INDEX)

        for row, (_, _, labels) in zip(label_batch, samples[:2]):
            assert torch.equal(row[: labels.size(0)], labels)
            assert torch.all(row[labels.size(0) :] == PAD_INDEX)

        # 생성한 batch를 Transformer에 그대로 입력할 수 있어야 함
        model = Transformer(
            vocab_size=tokenizer.vocab_size,
            d_model=16,
            num_heads=4,
            d_ff=32,
            num_layers=1,
            dropout=0.0,
            padding_index=tokenizer.pad_index,
        )
        logits, _, _, _ = model(source_batch, decoder_input_batch)
        assert logits.shape == (
            label_batch.size(0),
            label_batch.size(1),
            tokenizer.vocab_size,
        )

    print("TranslationDataset test passed")


def test_tokenizer() -> None:
    # 공백 정리 및 NFC 정규화 확인
    assert normalize_text("  Hello   world! \n ") == "Hello world!"

    raw_pairs = [
        ("  A man in an orange hat.  ", "  Ein Mann mit einem orangefarbenen Hut. "),
        ("", "Leere Quelle."),
        ("Empty target.", "   "),
        ("A Boston Terrier is running on lush green grass.", "Ein Boston Terrier läuft über saftig-grünes Gras."),
        ("A girl in karate uniform breaking a stick.", "Ein Mädchen im Karateanzug bricht einen Stock."),
    ]

    # 빈 문장 제거 및 단어 수 기준 길이 필터링 확인
    cleaned_pairs = preprocess_pairs(raw_pairs)
    assert len(cleaned_pairs) == 3
    assert len(preprocess_pairs(raw_pairs, max_length=7)) == 1

    # 공유 BPE 토크나이저 학습
    tokenizer = BPETokenizer()
    tokenizer.train_from_pairs(
        cleaned_pairs,
        vocab_size=120,
        min_frequency=1,
    )

    # Special token index가 Transformer의 padding_index(0) 및 약속된 상수와 일치해야 함
    assert tokenizer.pad_index == PAD_INDEX == 0
    assert tokenizer.unk_index == UNK_INDEX == 1
    assert tokenizer.bos_index == BOS_INDEX == 2
    assert tokenizer.eos_index == EOS_INDEX == 3
    assert tokenizer.vocab_size > 4

    # Source sequence는 [<tokens>, <eos>], Target sequence는 [<bos>, <tokens>, <eos>]
    source_text = "A girl in karate uniform breaking a stick."
    target_text = "Ein Mädchen im Karateanzug bricht einen Stock."

    source_ids = tokenizer.encode_source(source_text)
    target_ids = tokenizer.encode_target(target_text)

    assert source_ids[-1] == EOS_INDEX
    assert BOS_INDEX not in source_ids
    assert target_ids[0] == BOS_INDEX
    assert target_ids[-1] == EOS_INDEX

    # <bos>, <eos>, <pad> 및 <eos> 이후 토큰을 제외하고 원문으로 복원해야 함
    padded_target = torch.tensor(
        [*target_ids, PAD_INDEX, PAD_INDEX, 5],
        dtype=torch.long,
    )
    assert tokenizer.decode(source_ids) == source_text
    assert tokenizer.decode(padded_target) == target_text

    # 학습되지 않은 문자는 <unk>를 포함해 인코딩되어야 함
    unknown_ids = tokenizer.encode("🛸")
    assert UNK_INDEX in unknown_ids

    # 토크나이저 기준 max_length 필터링과 병렬 코퍼스 파일 로딩 및 저장/복원 확인
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        src_file = tmp_path / "train.en"
        tgt_file = tmp_path / "train.de"
        tok_file = tmp_path / "tokenizer.json"

        src_file.write_text(
            "A man in an orange hat.\n\nA girl in karate uniform breaking a stick.\n",
            encoding="utf-8",
        )
        tgt_file.write_text(
            "Ein Mann mit einem orangefarbenen Hut.\nIgnored line\nEin Mädchen im Karateanzug bricht einen Stock.\n",
            encoding="utf-8",
        )

        loaded_pairs_stream = load_parallel_corpus(src_file, tgt_file)
        assert iter(loaded_pairs_stream) is loaded_pairs_stream
        loaded_pairs = list(loaded_pairs_stream)
        assert len(loaded_pairs) == 2

        short_max_len = max(
            len(tokenizer.encode_source("A man in an orange hat.")),
            len(tokenizer.encode_target("Ein Mann mit einem orangefarbenen Hut.")),
        )
        filtered_by_tokens = list(
            load_parallel_corpus(
                src_file,
                tgt_file,
                max_length=short_max_len,
                tokenizer=tokenizer,
            )
        )
        assert filtered_by_tokens == [
            ("A man in an orange hat.", "Ein Mann mit einem orangefarbenen Hut.")
        ]

        # 두 파일의 줄 수가 다르면 스트림을 소비하는 시점에 오류가 발생해야 함
        mismatched_tgt_file = tmp_path / "mismatched.de"
        mismatched_tgt_file.write_text(
            "Ein Mann mit einem orangefarbenen Hut.\n",
            encoding="utf-8",
        )
        try:
            list(load_parallel_corpus(src_file, mismatched_tgt_file))
        except ValueError:
            pass
        else:
            raise AssertionError("Expected mismatched corpus lengths to fail")

        tokenizer.save(tok_file)
        reloaded = BPETokenizer.load(tok_file)
        assert reloaded.vocab_size == tokenizer.vocab_size
        assert reloaded.encode_target(target_text) == target_ids
        assert reloaded.decode(target_ids) == target_text

    # 토크나이저 출력을 Transformer에 바로 입력할 수 있는지 확인
    model = Transformer(
        vocab_size=tokenizer.vocab_size,
        d_model=16,
        num_heads=4,
        d_ff=32,
        num_layers=2,
        dropout=0.0,
        padding_index=tokenizer.pad_index,
    )
    source_batch = torch.tensor([source_ids], dtype=torch.long)
    decoder_input = torch.tensor([target_ids[:-1]], dtype=torch.long)
    logits, _, _, _ = model(source_batch, decoder_input)
    assert logits.shape == (1, len(target_ids) - 1, tokenizer.vocab_size)

    print("Tokenizer test passed")


def test_transformer() -> None:
    B = 2
    S_source = 5
    S_target = 4
    vocab_size = 30
    D_model = 16
    H = 4
    D_ff = 64
    N = 2

    model = Transformer(
        vocab_size=vocab_size,
        d_model=D_model,
        num_heads=H,
        d_ff=D_ff,
        num_layers=N,
        dropout=0.0,
        max_len=20,
        padding_index=0,
    )

    source_tokens = torch.tensor(
        [
            [1, 2, 3, 0, 0],
            [4, 5, 6, 7, 0],
        ]
    )
    target_tokens = torch.tensor(
        [
            [8, 9, 0, 0],
            [10, 11, 12, 0],
        ]
    )

    (
        logits,
        encoder_attention_weights,
        decoder_self_weights,
        decoder_cross_weights,
    ) = model(source_tokens, target_tokens)

    # 각 target 위치마다 vocabulary 전체에 대한 logit을 반환해야 함
    assert logits.shape == (B, S_target, vocab_size)
    assert len(encoder_attention_weights) == N
    assert len(decoder_self_weights) == N
    assert len(decoder_cross_weights) == N

    future_positions = torch.triu(
        torch.ones(S_target, S_target, dtype=torch.bool),
        diagonal=1,
    )

    for encoder_weights in encoder_attention_weights:
        assert encoder_weights.shape == (B, H, S_source, S_source)

        # Encoder는 source padding key를 보지 않아야 함
        assert torch.allclose(
            encoder_weights[0, :, :, 3:],
            torch.zeros(H, S_source, 2),
        )
        assert torch.allclose(
            encoder_weights[1, :, :, 4:],
            torch.zeros(H, S_source, 1),
        )

    for self_weights, cross_weights in zip(
        decoder_self_weights,
        decoder_cross_weights,
    ):
        assert self_weights.shape == (B, H, S_target, S_target)
        assert cross_weights.shape == (B, H, S_target, S_source)

        # Decoder Self-Attention은 미래 target token을 보지 않아야 함
        assert torch.allclose(
            self_weights[..., future_positions],
            torch.zeros(B, H, future_positions.sum()),
        )

        # Decoder Cross-Attention은 source padding key를 보지 않아야 함
        assert torch.allclose(
            cross_weights[0, :, :, 3:],
            torch.zeros(H, S_target, 2),
        )
        assert torch.allclose(
            cross_weights[1, :, :, 4:],
            torch.zeros(H, S_target, 1),
        )

    # 원논문처럼 두 embedding과 출력 projection의 weight를 공유해야 함
    encoder_embedding = model.encoder.token_embedding.embedding.weight
    decoder_embedding = model.decoder.token_embedding.embedding.weight
    output_projection = model.output_projection.weight
    assert encoder_embedding.data_ptr() == decoder_embedding.data_ptr()
    assert encoder_embedding.data_ptr() == output_projection.data_ptr()

    # 최종 출력에서 Encoder와 Decoder 전체로 gradient가 전달되는지 확인
    logits[..., 0].sum().backward()
    assert encoder_embedding.grad is not None
    assert (
        model.encoder.layers[0].self_attention.query_projection.weight.grad
        is not None
    )
    assert (
        model.decoder.layers[-1].cross_attention.query_projection.weight.grad
        is not None
    )

    print("Transformer test passed")


def test_decoder() -> None:
    B = 2
    S_target = 4
    S_source = 5
    vocab_size = 30
    D_model = 16
    H = 4
    D_ff = 64
    N = 3

    model = Decoder(
        vocab_size=vocab_size,
        d_model=D_model,
        num_heads=H,
        d_ff=D_ff,
        num_layers=N,
        dropout=0.0,
        max_len=20,
    )

    target_tokens = torch.tensor(
        [
            [1, 2, 0, 0],
            [3, 4, 5, 0],
        ]
    )
    source_tokens = torch.tensor(
        [
            [1, 2, 3, 0, 0],
            [4, 5, 6, 7, 0],
        ]
    )
    encoder_output = torch.randn(
        B,
        S_source,
        D_model,
        requires_grad=True,
    )

    self_attention_mask = create_causal_mask(S_target)
    self_attention_mask = self_attention_mask | create_padding_mask(
        target_tokens
    )
    cross_attention_mask = create_padding_mask(source_tokens)

    out, self_attention_weights, cross_attention_weights = model(
        target_tokens,
        encoder_output,
        self_attention_mask,
        cross_attention_mask,
    )

    # Decoder 출력과 layer별 attention weight 개수 확인
    assert out.shape == (B, S_target, D_model)
    assert len(self_attention_weights) == N
    assert len(cross_attention_weights) == N

    future_positions = torch.triu(
        torch.ones(S_target, S_target, dtype=torch.bool),
        diagonal=1,
    )

    for self_weights, cross_weights in zip(
        self_attention_weights,
        cross_attention_weights,
    ):
        assert self_weights.shape == (B, H, S_target, S_target)
        assert cross_weights.shape == (B, H, S_target, S_source)

        # 모든 layer에서 미래 target token을 보지 않아야 함
        assert torch.allclose(
            self_weights[..., future_positions],
            torch.zeros(B, H, future_positions.sum()),
        )

        # 모든 layer에서 source padding key를 보지 않아야 함
        assert torch.allclose(
            cross_weights[0, :, :, 3:],
            torch.zeros(H, S_target, 2),
        )
        assert torch.allclose(
            cross_weights[1, :, :, 4:],
            torch.zeros(H, S_target, 1),
        )

    # 각 DecoderLayer는 서로 다른 파라미터를 가져야 함
    assert model.layers[0] is not model.layers[1]
    assert (
        model.layers[0].self_attention.query_projection.weight.data_ptr()
        != model.layers[1].self_attention.query_projection.weight.data_ptr()
    )

    # embedding, encoder 출력, 처음/마지막 layer까지 gradient 전달 확인
    out[..., 0].sum().backward()
    assert model.token_embedding.embedding.weight.grad is not None
    assert encoder_output.grad is not None
    assert (
        model.layers[0].self_attention.query_projection.weight.grad
        is not None
    )
    assert (
        model.layers[-1].cross_attention.query_projection.weight.grad
        is not None
    )

    print("Decoder test passed")


def test_decoder_layer() -> None:
    B = 2
    S_target = 4
    S_source = 5
    D_model = 16
    H = 4
    D_ff = 64

    model = DecoderLayer(
        d_model=D_model,
        num_heads=H,
        d_ff=D_ff,
        dropout=0.0,
    )

    x = torch.randn(B, S_target, D_model, requires_grad=True)
    encoder_output = torch.randn(
        B,
        S_source,
        D_model,
        requires_grad=True,
    )

    target_tokens = torch.tensor(
        [
            [1, 2, 0, 0],
            [3, 4, 5, 0],
        ]
    )
    source_tokens = torch.tensor(
        [
            [1, 2, 3, 0, 0],
            [4, 5, 6, 7, 0],
        ]
    )

    # 미래 token과 target padding을 함께 가리는 self-attention mask
    self_attention_mask = create_causal_mask(S_target)
    self_attention_mask = self_attention_mask | create_padding_mask(
        target_tokens
    )
    cross_attention_mask = create_padding_mask(source_tokens)

    out, self_attention_weights, cross_attention_weights = model(
        x,
        encoder_output,
        self_attention_mask,
        cross_attention_mask,
    )

    # DecoderLayer 출력과 두 attention weight의 shape 확인
    assert out.shape == (B, S_target, D_model)
    assert self_attention_weights.shape == (B, H, S_target, S_target)
    assert cross_attention_weights.shape == (B, H, S_target, S_source)

    # self-attention은 미래 token을 보지 않아야 함
    future_positions = torch.triu(
        torch.ones(S_target, S_target, dtype=torch.bool),
        diagonal=1,
    )
    assert torch.allclose(
        self_attention_weights[..., future_positions],
        torch.zeros(B, H, future_positions.sum()),
    )

    # target padding key 위치에는 self-attention을 주지 않아야 함
    assert torch.allclose(
        self_attention_weights[0, :, :, 2:],
        torch.zeros(H, S_target, 2),
    )
    assert torch.allclose(
        self_attention_weights[1, :, :, 3:],
        torch.zeros(H, S_target, 1),
    )

    # source padding key 위치에는 cross-attention을 주지 않아야 함
    assert torch.allclose(
        cross_attention_weights[0, :, :, 3:],
        torch.zeros(H, S_target, 2),
    )
    assert torch.allclose(
        cross_attention_weights[1, :, :, 4:],
        torch.zeros(H, S_target, 1),
    )

    # 마지막 연산이 LayerNorm이므로 feature 평균은 0에 가까워야 함
    assert torch.allclose(
        out.mean(dim=-1),
        torch.zeros(B, S_target),
        atol=1e-6,
    )

    # 입력과 세 sublayer까지 gradient가 전달되는지 확인
    out[..., 0].sum().backward()
    assert x.grad is not None
    assert encoder_output.grad is not None
    assert model.self_attention.query_projection.weight.grad is not None
    assert model.cross_attention.query_projection.weight.grad is not None
    assert model.feed_forward.linear1.weight.grad is not None

    print("DecoderLayer test passed")


def test_encoder() -> None:
    B = 2
    S = 5
    vocab_size = 30
    D_model = 16
    H = 4
    D_ff = 64
    N = 3

    model = Encoder(
        vocab_size=vocab_size,
        d_model=D_model,
        num_heads=H,
        d_ff=D_ff,
        num_layers=N,
        dropout=0.0,
        max_len=20,
    )

    tokens = torch.tensor(
        [
            [1, 2, 3, 0, 0],
            [4, 5, 6, 7, 0],
        ]
    )
    padding_mask = create_padding_mask(tokens)

    out, attention_weights = model(tokens, padding_mask)

    # Encoder 출력 shape와 layer별 attention weight shape 확인
    assert out.shape == (B, S, D_model)
    assert len(attention_weights) == N
    for weights in attention_weights:
        assert weights.shape == (B, H, S, S)

        # padding key 위치에는 attention을 주지 않아야 함
        assert torch.allclose(
            weights[0, :, :, 3:],
            torch.zeros(H, S, 2),
        )
        assert torch.allclose(
            weights[1, :, :, 4:],
            torch.zeros(H, S, 1),
        )

    # 각 EncoderLayer는 서로 다른 파라미터를 가져야 함
    assert model.layers[0] is not model.layers[1]
    assert (
        model.layers[0].self_attention.query_projection.weight.data_ptr()
        != model.layers[1].self_attention.query_projection.weight.data_ptr()
    )

    # embedding과 처음/마지막 layer까지 gradient가 전달되는지 확인
    out[..., 0].sum().backward()
    assert model.token_embedding.embedding.weight.grad is not None
    assert (
        model.layers[0].self_attention.query_projection.weight.grad
        is not None
    )
    assert (
        model.layers[-1].self_attention.query_projection.weight.grad
        is not None
    )

    print("Encoder test passed")


def test_encoder_layer() -> None:
    B = 2
    S = 5
    D_model = 16
    H = 4
    D_ff = 64

    model = EncoderLayer(
        d_model=D_model,
        num_heads=H,
        d_ff=D_ff,
        dropout=0.0,
    )

    x = torch.randn(B, S, D_model, requires_grad=True)
    tokens = torch.tensor(
        [
            [1, 2, 3, 0, 0],
            [4, 5, 6, 7, 0],
        ]
    )
    padding_mask = create_padding_mask(tokens)

    out, attention_weights = model(x, padding_mask)

    # EncoderLayer는 입력과 같은 shape를 반환해야 함
    assert out.shape == (B, S, D_model)
    assert attention_weights.shape == (B, H, S, S)

    # padding key 위치에는 attention을 주지 않아야 함
    assert torch.allclose(
        attention_weights[0, :, :, 3:],
        torch.zeros(H, S, 2),
    )
    assert torch.allclose(
        attention_weights[1, :, :, 4:],
        torch.zeros(H, S, 1),
    )

    # 마지막 연산이 LayerNorm이므로 feature 평균은 0에 가까워야 함
    assert torch.allclose(
        out.mean(dim=-1),
        torch.zeros(B, S),
        atol=1e-6,
    )

    # 입력과 두 sublayer까지 gradient가 전달되는지 확인
    out.sum().backward()
    assert x.grad is not None
    assert model.self_attention.query_projection.weight.grad is not None
    assert model.feed_forward.linear1.weight.grad is not None

    print("EncoderLayer test passed")


def test_attention_masks() -> None:
    tokens = torch.tensor(
        [
            [1, 2, 0, 0],
            [3, 4, 5, 0],
        ]
    )

    padding_mask = create_padding_mask(tokens, padding_index=0)

    # (B, S) -> (B, 1, 1, S)
    assert padding_mask.shape == (2, 1, 1, 4)
    assert padding_mask.dtype == torch.bool
    assert torch.equal(
        padding_mask[:, 0, 0],
        torch.tensor(
            [
                [False, False, True, True],
                [False, False, False, True],
            ]
        ),
    )

    causal_mask = create_causal_mask(sequence_length=4)

    # 미래 token만 가리는 상삼각 mask여야 함
    assert causal_mask.shape == (1, 1, 4, 4)
    assert causal_mask.dtype == torch.bool
    assert torch.equal(
        causal_mask[0, 0],
        torch.tensor(
            [
                [False, True, True, True],
                [False, False, True, True],
                [False, False, False, True],
                [False, False, False, False],
            ]
        ),
    )

    # 두 mask는 broadcast되어 Decoder self-attention mask로 결합할 수 있어야 함
    combined_mask = padding_mask | causal_mask
    assert combined_mask.shape == (2, 1, 4, 4)

    print("Attention mask test passed")


def test_positionwise_feed_forward() -> None:
    B = 2
    S = 5
    D_model = 16
    D_ff = 64

    model = PositionwiseFeedForward(
        d_model=D_model,
        d_ff=D_ff,
    )

    x = torch.randn(B, S, D_model, requires_grad=True)
    out = model(x)

    # 입력과 출력 shape가 같아야 함
    assert out.shape == (B, S, D_model)

    # 중간 차원으로 확장한 뒤 다시 D_model로 줄여야 함
    assert model.linear1.weight.shape == (D_ff, D_model)
    assert model.linear2.weight.shape == (D_model, D_ff)

    # 같은 입력 vector에는 position과 관계없이 같은 결과가 나와야 함
    repeated = torch.randn(1, 1, D_model).expand(1, 2, -1)
    repeated_out = model(repeated)
    assert torch.allclose(repeated_out[:, 0], repeated_out[:, 1])

    # 입력과 두 Linear layer까지 gradient가 전달되는지 확인
    out.sum().backward()
    assert x.grad is not None
    assert model.linear1.weight.grad is not None
    assert model.linear2.weight.grad is not None

    print("PositionwiseFeedForward test passed")


def test_multi_head_attention() -> None:
    B = 2
    S_q = 3
    S_k = 5
    D_model = 16
    H = 4

    model = MultiHeadAttention(
        d_model=D_model,
        num_heads=H,
    )

    # query 길이와 key/value 길이가 다른 cross-attention 입력
    query = torch.randn(B, S_q, D_model, requires_grad=True)
    key = torch.randn(B, S_k, D_model, requires_grad=True)
    value = torch.randn(B, S_k, D_model, requires_grad=True)

    mask = torch.zeros(B, 1, S_q, S_k, dtype=torch.bool)
    mask[..., -1] = True

    out, attention_weights = model(query, key, value, mask)

    # 출력과 attention weight shape 확인
    assert out.shape == (B, S_q, D_model)
    assert attention_weights.shape == (B, H, S_q, S_k)

    # mask 처리된 위치에는 attention을 주지 않아야 함
    assert torch.allclose(
        attention_weights[..., -1],
        torch.zeros(B, H, S_q),
    )

    # 입력과 projection weight까지 gradient가 전달되는지 확인
    out.sum().backward()
    assert query.grad is not None
    assert key.grad is not None
    assert value.grad is not None
    assert model.query_projection.weight.grad is not None
    assert model.key_projection.weight.grad is not None
    assert model.value_projection.weight.grad is not None
    assert model.output_projection.weight.grad is not None

    # query, key, value가 같은 self-attention 입력도 처리할 수 있어야 함
    x = torch.randn(B, S_k, D_model)
    self_attention_out, self_attention_weights = model(x, x, x)

    assert self_attention_out.shape == (B, S_k, D_model)
    assert self_attention_weights.shape == (B, H, S_k, S_k)

    print("MultiHeadAttention test passed")


def test_scaled_dot_product_attention() -> None:
    B = 2
    H = 4
    S_q = 3
    S_k = 5
    D_head = 8

    model = ScaledDotProductAttention()

    query = torch.randn(B, H, S_q, D_head, requires_grad=True)
    key = torch.randn(B, H, S_k, D_head, requires_grad=True)
    value = torch.randn(B, H, S_k, D_head, requires_grad=True)

    # 마지막 key 위치를 모든 query에서 가림
    mask = torch.zeros(1, 1, S_q, S_k, dtype=torch.bool)
    mask[..., -1] = True

    out, attention_weights = model(query, key, value, mask)

    # 출력과 attention weight shape 확인
    assert out.shape == (B, H, S_q, D_head)
    assert attention_weights.shape == (B, H, S_q, S_k)

    # 각 query에 대한 attention weight의 합은 1이어야 함
    assert torch.allclose(
        attention_weights.sum(dim=-1),
        torch.ones(B, H, S_q),
    )

    # mask 처리된 위치에는 attention을 주지 않아야 함
    assert torch.allclose(
        attention_weights[..., -1],
        torch.zeros(B, H, S_q),
    )

    # query, key, value까지 gradient가 전달되는지 확인
    out.sum().backward()
    assert query.grad is not None
    assert key.grad is not None
    assert value.grad is not None

    # head 차원이 없는 단일 attention 입력도 처리할 수 있어야 함
    single_query = torch.randn(B, S_q, D_head)
    single_key = torch.randn(B, S_k, D_head)
    single_value = torch.randn(B, S_k, D_head)

    single_out, single_weights = model(
        single_query,
        single_key,
        single_value,
    )

    assert single_out.shape == (B, S_q, D_head)
    assert single_weights.shape == (B, S_q, S_k)

    print("ScaledDotProductAttention test passed")


def test_token_embedding() -> None:
    B = 2
    S = 4
    D = 16
    vocab_size = 50

    model = TokenEmbedding(
        vocab_size=vocab_size,
        d_model=D,
    )

    tokens = torch.tensor(
        [
            [1, 2, 3, 1],
            [4, 1, 0, 2],
        ],
        dtype=torch.long,
    )

    out = model(tokens)

    # 출력 shape 확인
    assert out.shape == (B, S, D)

    # embedding weight shape 확인
    assert model.embedding.weight.shape == (vocab_size, D)

    # 같은 token id는 같은 embedding vector를 반환해야 함
    assert torch.allclose(out[0, 0], out[0, 3])
    assert torch.allclose(out[0, 0], out[1, 1])

    # embedding weight까지 gradient가 전달되는지 확인
    out.sum().backward()
    assert model.embedding.weight.grad is not None
    assert model.embedding.weight.grad.shape == (vocab_size, D)

    print("TokenEmbedding test passed")


def test_positional_encoding() -> None:
    B = 4
    S = 10
    D = 16
    max_len = 100

    model = PositionalEncoding(
        d_model=D,
        dropout=0.0,
        max_len=max_len,
    )

    x = torch.zeros(B, S, D)
    out = model(x)

    # 출력 shape 확인
    assert out.shape == (B, S, D)

    # 저장된 positional encoding shape 확인
    assert model.pe.shape == (1, max_len, D)

    # x가 전부 0 -> 출력은 positional encoding과 같아야 함
    expected = model.pe[:, :S, :].expand(B, -1, -1)
    assert torch.allclose(out, expected)

    # position = 0 확인: sin(0) = 0, cos(0) = 1
    assert torch.allclose(
        model.pe[0, 0, 0::2],
        torch.zeros(D // 2),
    )
    assert torch.allclose(
        model.pe[0, 0, 1::2],
        torch.ones(D // 2),
    )

    print("PositionalEncoding test passed")


if __name__ == "__main__":
    test_train_epoch()
    test_training_setup()
    test_translation_dataset()
    test_tokenizer()
    test_transformer()
    test_decoder()
    test_decoder_layer()
    test_encoder()
    test_encoder_layer()
    test_attention_masks()
    test_positionwise_feed_forward()
    test_multi_head_attention()
    test_scaled_dot_product_attention()
    test_token_embedding()
    test_positional_encoding()
