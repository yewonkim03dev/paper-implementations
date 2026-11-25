import math
from typing import Literal

import torch
from torch import Tensor, nn


def _build_attention_mask(
    causal_mask: Tensor,
    padding_mask: Tensor,
    num_heads: int,
) -> Tensor:
    """causal mask와 padding mask를 결합하고 padding query 행의 all masked NaN을 방지"""
    # causal_mask (S, S) True 위치(미래) 참조 차단
    # padding_mask (B, S) True 위치(padding) 참조 차단
    seq_len = causal_mask.size(0)

    # (B, S, S) 미래 위치 또는 padding key 차단
    # causal_mask(S, S) -> (1, S, S), padding_mask(B, S) -> (B, 1, S)
    # 브로드캐스팅 + OR => (B, S, S), True 위치(미래 or padding) 참조 차단
    # 이 시점까지는 패딩 + causal 때문에 일부 행은 완전히 False -> 무시되는 행이 있을 수 있음
    attn_mask = causal_mask.unsqueeze(0) | padding_mask.unsqueeze(1)

    # 왼쪽 padding query는 모든 key가 막혀 softmax(-inf, ..., -inf) -> NaN이 될 수 있으니까
    # padding query 행에 한해 대각선 (t, t)만 열어둠 (출력은 블록 끝에서 0으로 마스킹됨)

    # 대각만 True인 단위행렬 생성, (1, S, S)
    diag_mask = torch.eye(seq_len, device=causal_mask.device, dtype=torch.bool).unsqueeze(0)

    # padding mask(B, S) -> (B, S, 1) -> 대각선 행렬이랑 원소별 AND
    # => AND연산에서 브로드캐스팅 발생하면서 padding mask값이 Key축을 따라 복사, AND로 차단을 해제해야 하는 위치만 True가 됨
    # NOT으로 뒤집고(차단 해제 셀만 F로 만듬) -> AND 적용 -> 대각선 칸만 False로 차단해제됨
    attn_mask = attn_mask & ~(padding_mask.unsqueeze(-1) & diag_mask)
    if num_heads == 1:
        return attn_mask
    # MultiheadAttention 3D attn_mask 규격: (B * num_heads, S, S)
    return attn_mask.repeat_interleave(num_heads, dim=0)


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

        attn_mask = _build_attention_mask(causal_mask, padding_mask, self.attention.num_heads)
        attn_output, _ = self.attention(query=query, key=key, value=value, attn_mask=attn_mask, need_weights=False)

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
        attn_mask = _build_attention_mask(causal_mask, padding_mask, self.attention.num_heads)
        attn_output, _ = self.attention(query=query, key=x, value=x, attn_mask=attn_mask, need_weights=False)

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

        # 논문/저자 코드 비교 시 초기화를 통제 변인으로 고정하고 초기 logit 포화를 방지
        # 논문 버전, 저자 코드 버전 둘다 xavier초기화 사용
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Embedding):
            nn.init.xavier_uniform_(module.weight)
            if module.padding_idx is not None:
                with torch.no_grad():
                    module.weight[module.padding_idx].zero_()
        elif isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.MultiheadAttention):
            if module.in_proj_weight is not None:
                nn.init.xavier_uniform_(module.in_proj_weight)
            if module.in_proj_bias is not None:
                nn.init.zeros_(module.in_proj_bias)

    def forward(self, item_ids: Tensor) -> Tensor:
        # item_ids: (B, S), 0으로 왼쪽 padding, 1 <= S <= max_len
        if item_ids.ndim != 2:
            raise ValueError(f"item_ids는 2차원 (B, S) 텐서여야 함: {tuple(item_ids.shape)}")

        # padding mask, causal mask
        B, S = item_ids.shape
        max_len = self.position_embedding.num_embeddings
        if S < 1 or S > max_len:
            raise ValueError(f"시퀀스 길이는 1 이상 {max_len} 이하여야 함: {S}")
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

    def score_items(self, features: Tensor, candidate_ids: Tensor) -> Tensor:
        # features: (B, S, D) 학습 / (B, D) 평가
        # candidate_ids: (B, S, K) 학습 / (B, K) 평가

        # (B, S, K, D) or (B, K, D)
        candidate_embeddings = self.item_embedding(candidate_ids)

        # (B, S, D) -> (B, S, 1, D) or (B, D) -> (B, 1, D)
        features = features.unsqueeze(-2)

        # (B, S, 1, D) @ (B, S, D, K) -> (B, S, 1, K) -> (B, S, K)
        scores = (features @ candidate_embeddings.transpose(-1, -2)).squeeze(-2)
        # return (B, S, K) 학습 / (B, K) 평가
        return scores
