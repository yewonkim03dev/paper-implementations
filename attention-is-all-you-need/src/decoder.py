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
    pass
