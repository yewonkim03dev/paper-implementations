from torch import Tensor, nn

from .attention import MultiHeadAttention
from .embedding import PositionalEncoding, TokenEmbedding
from .feed_forward import PositionwiseFeedForward


class EncoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.self_attention = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(
        self,
        x: Tensor,
        mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        # x: (B, S, D_model)

        # Self-Attention: (B, S, D_model) -> (B, S, D_model)
        attention_output, attention_weights = self.self_attention(
            x,
            x,
            x,
            mask,
        )

        # Post-LayerNorm: Residual -> LayerNorm
        # (B, S, D_model)
        x = self.norm1(x + self.dropout1(attention_output))

        # Feed Forward: (B, S, D_model) -> (B, S, D_model)
        feed_forward_output = self.feed_forward(x)

        # Post-LayerNorm: Residual -> LayerNorm
        # (B, S, D_model)
        x = self.norm2(x + self.dropout2(feed_forward_output))

        # return (B, S, D_model), (B, H, S, S)
        return x, attention_weights


class Encoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        num_heads: int,
        d_ff: int,
        num_layers: int,
        dropout: float = 0.1,
        max_len: int = 5000,
    ):
        super().__init__()

        if num_layers <= 0:
            raise ValueError("num_layers must be greater than 0")

        self.token_embedding = TokenEmbedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(
            d_model,
            dropout,
            max_len,
        )
        self.layers = nn.ModuleList(
            [
                EncoderLayer(d_model, num_heads, d_ff, dropout)
                for _ in range(num_layers)
            ]
        )

    def forward(
        self,
        tokens: Tensor,
        mask: Tensor | None = None,
    ) -> tuple[Tensor, list[Tensor]]:
        # tokens: (B, S)

        # Token Embedding: (B, S) -> (B, S, D_model)
        x = self.token_embedding(tokens)

        # Positional Encoding: (B, S, D_model) -> (B, S, D_model)
        x = self.positional_encoding(x)

        attention_weights = []
        for layer in self.layers:
            # EncoderLayer: (B, S, D_model) -> (B, S, D_model)
            x, layer_attention_weights = layer(x, mask)

            # 각 layer의 attention weight: (B, H, S, S)
            attention_weights.append(layer_attention_weights)

        # return (B, S, D_model), N * (B, H, S, S)
        return x, attention_weights
