import math
from typing import Literal

import torch
from torch import Tensor, nn


class PaperSASRecBlock(nn.Module):
    """논문 III-C의 self-attention block"""

    def __init__(
        self,
        hidden_dim: int = 50,
        num_heads: int = 1,
        dropout: float = 0.2,
        *,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()

        self.attention_norm = nn.LayerNorm(hidden_dim, eps=eps)
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=0.0,
            batch_first=True,
        )
        self.attention_output_dropout = nn.Dropout(dropout)

        self.ffn_norm = nn.LayerNorm(hidden_dim, eps=eps)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.ffn_output_dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: Tensor,
        causal_mask: Tensor,
        padding_mask: Tensor,
    ) -> Tensor:
        # x: (B, S, D)
        # causal_mask: (S, S), True 위치 참조 차단
        # padding_mask: (B, S), True 위치 padding

        # Q, K, V 정규화
        normalized_x = self.attention_norm(x)
        query = key = value = normalized_x

        attn_output, _ = self.attention(query=query, key=key, value=value, attn_mask=causal_mask, key_padding_mask=padding_mask, need_weights=False)

        # dropout(attn_output) + x residual
        x = x + self.attention_output_dropout(attn_output)

        # FFN 적용 이전 LayerNorm
        ffn_input = self.ffn_norm(x)
        ffn_output = self.ffn(ffn_input)

        # FFN 출력 dropout + residual conn
        x = x + self.ffn_output_dropout(ffn_output)

        # return (B, S, D)
        # padding 위치 출력 0 처리
        return x.masked_fill(padding_mask.unsqueeze(-1), 0.0)


class AuthorSASRecBlock(nn.Module):
    """저자 공개 코드(https://github.com/kang205/SASRec)를 참고한 self-attention block"""

    def __init__(
        self,
        hidden_dim: int = 50,
        num_heads: int = 1,
        dropout: float = 0.2,
        *,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()

        # attention LayerNorm
        self.attention_norm = nn.LayerNorm(hidden_dim, eps=eps)

        # MultiheadAttention(batch_first=True) (B, S, D)
        self.attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, dropout=dropout, batch_first=True)

        #  FFN LayerNorm
        self.ffn_norm = nn.LayerNorm(hidden_dim, eps=eps)

        # Linear -> ReLU -> Dropout -> Linear -> Dropout
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Dropout(dropout)
        )

    def forward(
        self,
        x: Tensor,
        causal_mask: Tensor,
        padding_mask: Tensor,
    ) -> Tensor:
        # x: (B, S, D)
        # causal_mask: (S, S), True 위치 참조 차단
        # padding_mask: (B, S), True 위치 padding

        # Q Norm - (B, S, D)
        query = self.attention_norm(x)

        # Q = 정규화 입력, K,V = 기존 입력
        attn_output, _ = self.attention(query=query, key=x, value=x, attn_mask=causal_mask, key_padding_mask=padding_mask, need_weights=False)

        # 정규화 Q와 residual conn
        x = query + attn_output

        # FFN 앞 LayerNorm과 정규화한 입력 residual conn
        ffn_input = self.ffn_norm(x)
        x = ffn_input + self.ffn(ffn_input)
        # padding 위치 출력 0
        # return (B, S, D)
        return x.masked_fill(padding_mask.unsqueeze(-1), 0.0)


class SASRec(nn.Module):
    """Item/position embedding과 SASRec block stack"""

    def __init__(
        self,
        num_items: int,
        max_len: int = 200,
        hidden_dim: int = 50,
        num_blocks: int = 2,
        num_heads: int = 1,
        dropout: float = 0.2,
        *,
        block_variant: Literal["paper", "author"] = "author",
        eps: float = 1e-8,
        final_norm_eps: float | None = None,
    ) -> None:
        super().__init__()

        # num_items: padding을 제외한 item 수, 실제 item ID 1 ~ num_items
        # 임베딩 테이블에서 패딩 전용으로 쓸 인덱스 = 0. 아이템 시퀀스에서 0인 부분은 패딩처리
        self.item_embedding = nn.Embedding(num_embeddings=num_items + 1, embedding_dim=hidden_dim, padding_idx=0)

        # 학습 가능한 position Embedding(max_len, hidden_dim)
        self.position_embedding = nn.Embedding(max_len, hidden_dim)

        # 임베딩 드롭아웃
        self.embedding_dropout = nn.Dropout(dropout)

        if block_variant == "paper":
            block_class = PaperSASRecBlock
        elif block_variant == "author":
            block_class = AuthorSASRecBlock
        else:
            raise ValueError(f"알 수 없는 SASRec block 설정값: {block_variant}")

        self.blocks = nn.ModuleList(
            block_class(hidden_dim, num_heads, dropout, eps=eps) for _ in range(num_blocks)
        )

        # 마지막 LayerNorm
        self.norm = nn.LayerNorm(normalized_shape=hidden_dim, eps=eps if final_norm_eps is None else final_norm_eps,)
