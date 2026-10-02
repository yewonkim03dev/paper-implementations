"""Local validation entry point for the Attention Is All You Need implementation."""

import torch

from src.embedding import PositionalEncoding


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
    test_positional_encoding()
