import torch
import torch.nn.functional as F
from torch import nn

from settings import settings


class FFN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.l1 = nn.Linear(
            settings.embedding_dim, 4 * settings.embedding_dim, bias=False
        )
        self.l2 = nn.Linear(
            4 * settings.embedding_dim, settings.embedding_dim, bias=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.l1(x)
        x = F.relu(x)
        x = self.l2(x)
        return x
