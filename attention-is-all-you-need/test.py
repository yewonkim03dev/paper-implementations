"""Local validation entry point for the Attention Is All You Need implementation."""

import torch

from src.attention import MultiHeadAttention, ScaledDotProductAttention
from src.decoder import DecoderLayer
from src.embedding import PositionalEncoding, TokenEmbedding
from src.encoder import Encoder, EncoderLayer
from src.feed_forward import PositionwiseFeedForward
from src.masks import create_causal_mask, create_padding_mask


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
    test_decoder_layer()
    test_encoder()
    test_encoder_layer()
    test_attention_masks()
    test_positionwise_feed_forward()
    test_multi_head_attention()
    test_scaled_dot_product_attention()
    test_token_embedding()
    test_positional_encoding()
