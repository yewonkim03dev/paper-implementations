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

        # query @ key^T
        # (B, H, S_q, D_head) @ (B, H, D_head, S_k)
        # -> (B, H, S_q, S_k)
        scores = torch.matmul(query, key.transpose(-2, -1))
        scores = scores / math.sqrt(d_k)

        if mask is not None:
            # mask: (B, H, S_q, S_k)로 broadcast 가능한 shape
            scores = scores.masked_fill(mask, float("-inf"))

        # (B, H, S_q, S_k)
        attention_weights = torch.softmax(scores, dim=-1)

        # attention_weights @ value
        # (B, H, S_q, S_k) @ (B, H, S_k, D_head)
        # -> (B, H, S_q, D_head)
        output = torch.matmul(attention_weights, value)

        # return (B, H, S_q, D_head), (B, H, S_q, S_k)
        return output, attention_weights


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()

        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads

        self.query_projection = nn.Linear(d_model, d_model)
        self.key_projection = nn.Linear(d_model, d_model)
        self.value_projection = nn.Linear(d_model, d_model)
        self.output_projection = nn.Linear(d_model, d_model)

        self.attention = ScaledDotProductAttention()

    def _split_heads(self, x: Tensor) -> Tensor:
        # (B, S, D_model)
        B, S, _ = x.shape

        # (B, S, D_model) -> (B, S, H, D_head) -> (B, H, S, D_head)
        return x.view(B, S, self.num_heads, self.d_head).transpose(1, 2)

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        # query: (B, S_q, D_model)
        # key, value: (B, S_k, D_model)
        B, S_q, _ = query.shape

        # Linear projection: (B, S, D_model) -> (B, S, D_model)
        query = self.query_projection(query)
        key = self.key_projection(key)
        value = self.value_projection(value)

        # Head 분할: (B, S, D_model) -> (B, H, S, D_head)
        query = self._split_heads(query)
        key = self._split_heads(key)
        value = self._split_heads(value)

        # (B, H, S_q, D_head), (B, H, S_q, S_k)
        attention_output, attention_weights = self.attention(
            query,
            key,
            value,
            mask,
        )

        # Head 결합
        # (B, H, S_q, D_head) -> (B, S_q, H, D_head) -> (B, S_q, D_model)
        attention_output = attention_output.transpose(1, 2).contiguous()
        attention_output = attention_output.view(B, S_q, self.d_model)

        # (B, S_q, D_model)
        output = self.output_projection(attention_output)

        # return (B, S_q, D_model), (B, H, S_q, S_k)
        return output, attention_weights
