import torch
from torch import Tensor


def create_padding_mask(tokens: Tensor, padding_index: int = 0) -> Tensor:
    # tokens: (B, S_k)

    # (B, S_k) -> (B, 1, 1, S_k)
    # True인 key 위치는 attention에서 제외한다.
    return (tokens == padding_index).unsqueeze(1).unsqueeze(2)


def create_causal_mask(
    sequence_length: int,
    device: torch.device | None = None,
) -> Tensor:
    if sequence_length <= 0:
        raise ValueError("sequence_length must be greater than 0")

    # 현재 위치보다 뒤에 있는 token을 True로 표시한다.
    # (S, S)
    mask = torch.triu(
        torch.ones(
            sequence_length,
            sequence_length,
            dtype=torch.bool,
            device=device,
        ),
        diagonal=1,
    )

    # (S, S) -> (1, 1, S, S)
    return mask.unsqueeze(0).unsqueeze(0)
