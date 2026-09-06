from pathlib import Path

import torch
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    source_datafile: Path = Path("../data/tiny_stories.txt")
    train_file: Path = Path("../data/tiny_stories_train.txt")
    val_file: Path = Path("../data/tiny_stories_val.txt")
    sft_file: Path = Path("../data/sft_100.json")
    context_length: int = 32
    embedding_dim: int = 32
    num_heads: int = 2
    tile_size: int = 2
    device: torch.device = torch.device("mps" if torch.mps.is_available() else "cpu")
    num_trf_blocks: int = 1
    epochs: int = 100

    batch_size: int = 32
    train_split: float = 0.9


settings = Settings()
