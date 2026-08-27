from typing import cast

import torch

from gpt import AbbyGPT
from my_tokenizer import MyCharTokenizer
from settings import settings


class Inference:
    def __init__(self, model: AbbyGPT, tokenizer: MyCharTokenizer) -> None:
        self.model = model
        self.tokenizer = tokenizer

    def pre_training(self, x: str, num_token_to_predict: int = 10) -> str:
        if self.model.training:
            self.model.eval()

        probs = self.model(
            cast(torch.Tensor, self.tokenizer.encode(x, return_pt=True)).to(
                settings.device
            )
        )
        preds = torch.argmax(probs, dim=-1)
        op = self.tokenizer.decode(preds.flatten().tolist())
        return op
