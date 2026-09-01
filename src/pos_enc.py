import math

import torch
from torch import nn

from settings import settings


class PositionalEncoding(nn.Module):
    def __init__(
        self,
    ) -> None:
        super().__init__()
        pos_encoding = torch.zeros(settings.context_length, settings.embedding_dim)
        div_term = torch.exp(
            -1
            * torch.arange(0, settings.embedding_dim, 2)
            * math.log(1000)
            / settings.embedding_dim
        )
        posn = torch.arange(settings.context_length).reshape(-1, 1)

        pos_encoding[:, 0::2] = torch.sin(posn * div_term)
        pos_encoding[:, 1::2] = torch.cos(posn * div_term)
        self.register_buffer("pos_encoding", pos_encoding, persistent=False)

    def forward(self, x: torch.Tensor):
        b, length, _ = x.shape
        return x + self.pos_encoding[0:length, :]  # type: ignore


if __name__ == "__main__":
    x = torch.randn((1, settings.context_length, settings.embedding_dim))
    print("x.shape ", x.shape)
    xx = PositionalEncoding()
    op = xx(x)
    print(op)
    print(op.shape)
