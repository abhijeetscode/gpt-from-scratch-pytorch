import torch
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    context_length: int = 16
    embedding_dim: int = 128
    num_heads: int = 4
    tile_size: int = 2
    device: torch.device = torch.device("mps" if torch.mps.is_available() else "cpu")
    num_trf_blocks: int = 12
    epochs: int = 5

    batch_size: int = 32


settings = Settings()
