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
        self.block_variant = block_variant

        self.blocks = nn.ModuleList(
            block_class(hidden_dim, num_heads, dropout, eps=eps) for _ in range(num_blocks)
        )

        # 마지막 LayerNorm
        self.norm = nn.LayerNorm(normalized_shape=hidden_dim, eps=eps if final_norm_eps is None else final_norm_eps,)

    def forward(self, item_ids: Tensor) -> Tensor:
        # item_ids: (B, S), 0으로 왼쪽 padding, S <= max_len

        # padding mask, causal mask
        B, S = item_ids.shape
        # (B, S)
        padding_mask = (item_ids == 0)
        # (S, S)
        causal_mask = torch.triu(torch.ones(S, S, device=item_ids.device, dtype=torch.bool), diagonal=1)

        # item embedding << sqrt(hidden_dim) 스케일링 (B, S) -> (B, S, D)
        # 저자 코드에서는 임베딩에 sqrt(d_model) 을 곱함. 저자 코드 모드일때만 곱해줌 (아마도 트랜스포머 관례를 따른것 같다_트랜스포머 논문 3.4)
        x = self.item_embedding(item_ids) * (math.sqrt(self.item_embedding.embedding_dim) if self.block_variant == "author" else 1.0)

        # position embedding 합산과 dropout
        # zero-base tensor idx
        # max_len만큼 left padding case(고정 길이), 배치 내 가장 긴 시퀀스만큼만 left padding case(가변 길이) 지원 = (max_len - S, max_len)
        # position id는 max_len 기준 오른쪽 정렬
        max_len = self.position_embedding.num_embeddings
        position_ids = torch.arange(max_len - S, max_len, device=item_ids.device)
        x = self.embedding_dropout(x + self.position_embedding(position_ids))

        # padding 위치 출력 0으로 마스킹
        x = x.masked_fill(padding_mask.unsqueeze(-1), 0.0)

        # self.blocks 순차 적용 (B, S, D) -> (B, S, D)
        for block in self.blocks:
            x =  block(x, causal_mask=causal_mask, padding_mask=padding_mask)
        # 마지막 LayerNorm
        x = self.norm(x)

        # padding 위치 출력 0 처리
        # return (B, S, D)
        return x.masked_fill(padding_mask.unsqueeze(-1), 0.0)
