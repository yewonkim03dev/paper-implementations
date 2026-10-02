"""Local validation entry point for the Attention Is All You Need implementation."""

import torch

from src.embedding import PositionalEncoding, TokenEmbedding


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
    test_token_embedding()
    test_positional_encoding()
