import math

import torch
from torch import Tensor, nn


class ScaledDotProductAttention(nn.Module):
    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        # query: (B, H, S_q, D_head)
        # key: (B, H, S_k, D_head)
        # value: (B, H, S_k, D_head)
        d_k = query.size(-1)

        # scores: (B, H, S_q, S_k)
        scores = torch.matmul(query, key.transpose(-2, -1))
        scores = scores / math.sqrt(d_k)

        if mask is not None:
            scores = scores.masked_fill(mask, float("-inf"))

        # (B, H, S_q, S_k)
        attention_weights = torch.softmax(scores, dim=-1)

        # (B, H, S_q, D_head)
        output = torch.matmul(attention_weights, value)

        # return (B, H, S_q, D_head), (B, H, S_q, S_k)
        return output, attention_weights


class MultiHeadAttention(nn.Module):
    pass
