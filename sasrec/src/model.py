from torch import Tensor, nn


class PaperSASRecBlock(nn.Module):
    """논문 III-C의 self-attention block"""

    def __init__(
        self,
        hidden_dim: int = 50,
        num_heads: int = 1,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()

        self.attention_norm = nn.LayerNorm(hidden_dim, eps=1e-8)
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=0.0,
            batch_first=True,
        )
        self.attention_output_dropout = nn.Dropout(dropout)

        self.ffn_norm = nn.LayerNorm(hidden_dim, eps=1e-8)
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
    ) -> None:
        super().__init__()

        # attention LayerNorm
        self.attention_norm = nn.LayerNorm(hidden_dim, eps=1e-8)

        # MultiheadAttention(batch_first=True) (B, S, D)
        self.attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, dropout=dropout, batch_first=True)

        #  FFN LayerNorm
        self.ffn_norm = nn.LayerNorm(hidden_dim, eps=1e-8)

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
