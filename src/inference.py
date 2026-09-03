from settings import settings


def model_inference(
    model, tokenizer, x: str, num_token_to_predict: int = 1
) -> list[int]:
    model.to(settings.device)
    input_tokens = tokenizer.encode(x).ids
    current_length = len(input_tokens)

    if current_length > settings.context_length:
        raise ValueError(
            f"Input token length ({len(input_tokens)}) exceeds "
            f"context length ({settings.context_length})"
        )

    tokens = torch.zeros(
        current_length + num_token_to_predict,
        dtype=torch.long,
        device=settings.device,
    )
    tokens[:current_length] = torch.tensor(
        input_tokens, dtype=torch.long, device=settings.device
    )
    model.eval()

    for _ in range(num_token_to_predict):
        logits = model(tokens[:current_length].unsqueeze(0))
        pred_token = torch.argmax(logits[:, -1, :], dim=-1).item()
        tokens[current_length] = pred_token
        current_length += 1
    op = tokenizer.decode(tokens.tolist())

    return op


if __name__ == "__main__":
    import torch
    from tokenizers import Tokenizer

    from gpt import AbbyGPT

    my_tokenizer = Tokenizer.from_file("./bpe_tokenizer.json")
    abbygpt = AbbyGPT(vocab_size=my_tokenizer.get_vocab_size())
    abbygpt.load_state_dict(torch.load("./AbbyGPT_StateDict.pt"))

    op = model_inference(
        model=abbygpt, tokenizer=my_tokenizer, x="I am", num_token_to_predict=10
    )
    print(op)
