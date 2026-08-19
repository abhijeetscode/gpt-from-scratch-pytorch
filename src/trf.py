import torch
from torch import nn

from ffn import FFN
from flash_attention import MultiHeadAttention
from rms_norm import RMSNorm
from settings import settings


class TransformerBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.attn_block = MultiHeadAttention().to(settings.device)
        self.rms_norm = RMSNorm().to(settings.device)
        self.ffn = FFN().to(settings.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_raw = x.clone()
        x = self.rms_norm(x)  # pre-norm
        x = self.attn_block(x)
        x = x + x_raw

        x_raw = x.clone()
        x = self.rms_norm(x)
        x = self.ffn(x)
        return x + x_raw


if __name__ == "__main__":
    input_tensor = torch.randn((1, settings.context_length, settings.embedding_dim)).to(
        settings.device
    )
    trf = TransformerBlock().to(settings.device)
    op = trf(input_tensor)
    print(op.shape)
