from torch import nn

from .attention import MultiHeadAttention
from .embedding import PositionalEncoding, TokenEmbedding
from .feed_forward import PositionwiseFeedForward

class DecoderLayer(nn.Module):
    pass

class Decoder(nn.Module):
    pass