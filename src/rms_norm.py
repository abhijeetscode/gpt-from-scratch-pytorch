import torch
from torch import nn

from settings import settings


class RMSNorm(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x / torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True))


if __name__ == "__main__":
    op = RMSNorm()(x=torch.randn(settings.context_length, settings.embedding_dim))
    print(op.shape)
