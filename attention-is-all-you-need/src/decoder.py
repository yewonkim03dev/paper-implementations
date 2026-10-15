from torch import Tensor, nn

from .attention import MultiHeadAttention
from .embedding import PositionalEncoding, TokenEmbedding
from .feed_forward import PositionwiseFeedForward


class DecoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.self_attention = MultiHeadAttention(d_model, num_heads)
        self.cross_attention = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

    def forward(
        self,
        x: Tensor,
        encoder_output: Tensor,
        self_attention_mask: Tensor | None = None,
        cross_attention_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        # x: (B, S_target, D_model)
        # encoder_output: (B, S_source, D_model)

        # Masked Self-Attention
        # Q, K, V: x -> (B, S_target, D_model)
        self_attention_output, self_attention_weights = self.self_attention(
            x,
            x,
            x,
            self_attention_mask,
        )

        # Post-LayerNorm: Residual -> LayerNorm
        # (B, S_target, D_model)
        x = self.norm1(x + self.dropout1(self_attention_output))

        # Encoder-Decoder Attention
        # Q: x (B, S_target, D_model)
        # K, V: encoder_output (B, S_source, D_model)
        cross_attention_output, cross_attention_weights = self.cross_attention(
            x,
            encoder_output,
            encoder_output,
            cross_attention_mask,
        )

        # Post-LayerNorm: Residual -> LayerNorm
        # (B, S_target, D_model)
        x = self.norm2(x + self.dropout2(cross_attention_output))

        # Feed Forward: (B, S_target, D_model) -> (B, S_target, D_model)
        feed_forward_output = self.feed_forward(x)

        # Post-LayerNorm: Residual -> LayerNorm
        # (B, S_target, D_model)
        x = self.norm3(x + self.dropout3(feed_forward_output))

        # return (B, S_target, D_model),
        #        (B, H, S_target, S_target),
        #        (B, H, S_target, S_source)
        return x, self_attention_weights, cross_attention_weights


class Decoder(nn.Module):
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
                DecoderLayer(d_model, num_heads, d_ff, dropout)
                for _ in range(num_layers)
            ]
        )

    def forward(
        self,
        tokens: Tensor,
        encoder_output: Tensor,
        self_attention_mask: Tensor | None = None,
        cross_attention_mask: Tensor | None = None,
    ) -> tuple[Tensor, list[Tensor], list[Tensor]]:
        # tokens: (B, S_target)
        # encoder_output: (B, S_source, D_model)

        # Token Embedding: (B, S_target) -> (B, S_target, D_model)
        x = self.token_embedding(tokens)

        # Positional Encoding
        # (B, S_target, D_model) -> (B, S_target, D_model)
        x = self.positional_encoding(x)

        self_attention_weights = []
        cross_attention_weights = []

        for layer in self.layers:
            # DecoderLayer: (B, S_target, D_model)
            # -> (B, S_target, D_model)
            x, layer_self_weights, layer_cross_weights = layer(
                x,
                encoder_output,
                self_attention_mask,
                cross_attention_mask,
            )

            # Self-Attention: (B, H, S_target, S_target)
            self_attention_weights.append(layer_self_weights)

            # Cross-Attention: (B, H, S_target, S_source)
            cross_attention_weights.append(layer_cross_weights)

        # return (B, S_target, D_model),
        #        N * (B, H, S_target, S_target),
        #        N * (B, H, S_target, S_source)
        return x, self_attention_weights, cross_attention_weights
