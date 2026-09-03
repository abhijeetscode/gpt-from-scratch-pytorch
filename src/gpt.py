import torch
from torch import nn

from pos_enc import PositionalEncoding
from rms_norm import RMSNorm
from settings import settings
from trf import TransformerBlock


class AbbyGPT(nn.Module):
    def __init__(self, vocab_size: int) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.trf_blocks = nn.ModuleList(
            [
                TransformerBlock().to(settings.device)
                for _ in range(settings.num_trf_blocks)
            ]
        )
        self.l1 = nn.Linear(settings.embedding_dim, self.vocab_size)
        self.rms_norm = RMSNorm().to(settings.device)
        self.embedding_layer = nn.Embedding(self.vocab_size, settings.embedding_dim)
        self.posn_encodings = PositionalEncoding().to(settings.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.ndim == 2, "AbbyGPT expects 2D input matrix"
        x = self.embedding_layer(x)
        x = self.posn_encodings(x)
        x = self.rms_norm(x)
        for block in self.trf_blocks:
            x = block(x)
        x = self.rms_norm(x)
        x = self.l1(x)
        return x


if __name__ == "__main__":
    pass
    # input_tensor = torch.randn((1, settings.context_length, settings.embedding_dim), device=settings.device)
    # gpt = AbbyGPT(12).to(settings.device)
    # op = gpt(input_tensor)
    # assert op.shape == (1, settings.context_length, settings.vocab_size)
