import torch

from settings import settings

if __name__ == "__main__":
    from tokenizers import Tokenizer

    from gpt import AbbyGPT

    tokenizer = Tokenizer.from_file("./bpe_tokenizer.json")
    text = "I Love"
    enc = tokenizer.encode(text)
    # prompt guard
    if len(enc.ids) > settings.context_length:
        raise ValueError(
            f"prompt is {len(enc.ids)} tokens, context_length is {settings.context_length}"
        )
    token_indices = (
        torch.Tensor(enc.ids).to(dtype=torch.long, device=settings.device).unsqueeze(0)
    )

    state_dict = torch.load("./AbbyGPT.pt")
    model = AbbyGPT(vocab_size=tokenizer.get_vocab_size())
    model.load_state_dict(state_dict)
    model = model.to(settings.device)
    model.eval()
    with torch.no_grad():
        logits = model(token_indices)
        next_token_index = torch.argmax(logits[:, -1, :], dim=1).tolist()
        next_token = tokenizer.decode(next_token_index)
        print(next_token)
