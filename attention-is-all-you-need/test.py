"""Local validation entry point for the Attention Is All You Need implementation."""

import torch

from src.attention import MultiHeadAttention, ScaledDotProductAttention
from src.embedding import PositionalEncoding, TokenEmbedding


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
    test_multi_head_attention()
    test_scaled_dot_product_attention()
    test_token_embedding()
    test_positional_encoding()
