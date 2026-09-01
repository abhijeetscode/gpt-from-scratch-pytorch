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
        x_token_indices: torch.Tensor = (
            cast(torch.Tensor, self.tokenizer.encode(x, return_pt=True))
            .to(settings.device)
            .unsqueeze(0)
        )
        if len(x_token_indices) > settings.context_length:
            raise Exception("Input is out of context")  # noqa

        tokens: torch.Tensor = torch.zeros(
            len(x_token_indices) + num_token_to_predict,
            dtype=x_token_indices.dtype,
            device=x_token_indices.device,
        )
        current_initial_length: int = len(x_token_indices)
        tokens[:current_initial_length] = x_token_indices
        with torch.no_grad():
            for _ in range(num_token_to_predict):
                start_index: int = max(
                    current_initial_length - settings.context_length, 0
                )

                logits = self.model(tokens[start_index:current_initial_length])
                print("Logit shape :: ", logits.shape)
                pred_token_index: int = int(torch.argmax(logits[:, -1, :]).item())
                tokens[current_initial_length] = pred_token_index
                current_initial_length += 1
        return self.tokenizer.decode(tokens[:current_initial_length].tolist())
