import math

import torch
from torch import nn

from settings import settings


class FlashAttention(nn.Module):
    def __init__(self, head_dim: int) -> None:
        super().__init__()

        # Inputs
        self.context_length = settings.context_length
        self.embedding_dim = settings.embedding_dim
        self.head_dim = head_dim
        self.tile_size = settings.tile_size

        assert self.context_length % self.tile_size == 0, (
            "context length should be divisible by tile size"
        )

        # Layers
        self.W_q = nn.Linear(self.embedding_dim, self.head_dim, bias=False)
        self.W_k = nn.Linear(self.embedding_dim, self.head_dim, bias=False)
        self.W_v = nn.Linear(self.embedding_dim, self.head_dim, bias=False)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # context length is L
        # embedding dim is E
        # head_dim is D
        # tile size is T
        length: int = x.shape[1]
        q = self.W_q(x)  # (L, D)
        k = self.W_k(x)  # (L, D)
        v = self.W_v(x)  # (L, D)

        output_buffer = torch.zeros(
            (x.shape[0], length, self.head_dim), device=x.device
        )

        # create a range of context length with step size is tile size
        for i in range(0, length, self.tile_size):
            q_tile = q[..., i : i + self.tile_size, :]  # (T, D)
            running_max = torch.full((self.tile_size, 1), -math.inf, device=x.device)
            running_denom = torch.zeros((self.tile_size, 1), device=x.device)
            running_nume = torch.zeros((self.tile_size, self.head_dim), device=x.device)
            # one query tile will multiply with each key tile
            # one query tile multiple with key tiles in tile by tile. at the end one query tile will multiply each key tile but in the inner for loop we are doing step by step
            # not at once
            # at then end we need attention scores of each position with respect every other position
            # in this code we are doing tile by tile
            for j in range(0, self.context_length, self.tile_size):
                # need this, because we dont want to see future token
                # therefore we are contunuing
                if j > i:
                    continue
                k_tile = k[..., j : j + self.tile_size, :]  # (T, D)
                v_tile = v[..., j : j + self.tile_size, :]
                attn_scores = (
                    q_tile @ k_tile.transpose(-2, -1) / math.sqrt(self.head_dim)
                )  #  (T, T)
                if i == j:
                    attn_scores += mask
                new_max = torch.maximum(
                    running_max, torch.amax(attn_scores, dim=-1, keepdim=True)
                )
                alpha = torch.exp(running_max - new_max)
                running_denom = alpha * running_denom + torch.sum(
                    torch.exp(attn_scores - new_max), dim=-1, keepdim=True
                )
                running_nume = alpha * running_nume + (
                    torch.exp(attn_scores - new_max) @ v_tile
                )
                running_max = new_max
            output_buffer[..., i : i + self.tile_size, :] = running_nume / running_denom

        return output_buffer


class MultiHeadAttention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        assert settings.embedding_dim % settings.num_heads == 0, (
            "embedding dim should be divisible by num heads"
        )

        assert settings.context_length % settings.tile_size == 0, (
            "context length should be divisible by tile size"
        )

        self.head_dim: int = settings.embedding_dim // settings.num_heads
        self.tile_size = settings.tile_size

        self.heads = nn.ModuleList(
            [FlashAttention(self.head_dim) for _ in range(settings.num_heads)]
        )

        mask = torch.ones((self.tile_size, self.tile_size))
        mask = torch.tril(mask)
        mask = torch.where(mask.bool(), 0, -math.inf)
        self.register_buffer("mask", mask, persistent=False)

    def forward(self, x: torch.Tensor):
        x = torch.cat([head(x, self.mask) for head in self.heads], dim=-1)
        assert x.shape[-1] == settings.embedding_dim, (
            "feature dimension of output of MultiHeadAttention should match with embedding dim"
        )
        return x


if __name__ == "__main__":
    # considering 2 dimensions only context_length, embedding_dim
    x = torch.randn(
        (settings.context_length, settings.embedding_dim), device=settings.device
    )
    module = MultiHeadAttention().to(settings.device)
    op = module(x)
