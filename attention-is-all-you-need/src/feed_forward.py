from torch import Tensor, nn


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()

        self.linear1 = nn.Linear(d_model, d_ff)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x: Tensor) -> Tensor:
        # x: (B, S, D_model)

        # (B, S, D_model) -> (B, S, D_ff)
        x = self.linear1(x)
        x = self.relu(x)

        # (B, S, D_ff) -> (B, S, D_model)
        x = self.linear2(x)

        # return (B, S, D_model)
        return x
