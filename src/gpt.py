import torch 
import torch.nn as nn
from trf import TransformerBlock
from settings import settings
import torch.nn.functional as F
from rms_norm import RMSNorm

class AbbyGPT(nn.Module):
    def __init__(self, ) -> None:
        super().__init__()
        self.trf_blocks = [TransformerBlock().to(settings.device) for _ in range(settings.num_trf_blocks)]
        self.l1 = nn.Linear(settings.embedding_dim, settings.vocab_size)
        self.rms_norm = RMSNorm().to(settings.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.rms_norm(x)
        for block in self.trf_blocks:
            x = block(x)
        x = self.rms_norm(x)
        x = self.l1(x)
        x = F.softmax(x, dim=-1)
        return x

if __name__ == "__main__":
    input_tensor = torch.randn((1, settings.context_length, settings.embedding_dim), device=settings.device)
    gpt = AbbyGPT().to(settings.device)
    op = gpt(input_tensor)
    assert op.shape == (1, settings.context_length, settings.vocab_size)
    
    