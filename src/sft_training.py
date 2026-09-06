if __name__ == "__main__":
    import json

    import torch
    from tokenizers import Tokenizer

    from gpt import AbbyGPT
    from settings import settings

    tokenizer = Tokenizer.from_file("./bpe_tokenizer.json")
    data = json.loads(settings.sft_file.read_text(encoding="utf-8"))
    model = AbbyGPT(vocab_size=tokenizer.get_vocab_size())
    model.load_state_dict(torch.load("./AbbyGPT_StateDict.pt"))
    model.to(settings.device)
    model.eval()
    optimzer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_func = torch.nn.CrossEntropyLoss()
    for item in data:
        ip_string = item["user"]
        op = item["user"]

        ip_token_ids = torch.tensor(
            tokenizer.encode(ip_string).ids, dtype=torch.long, device=settings.device
        ).unsqueeze(0)
        op_token_ids = tokenizer.encode(ip_string).ids
        num_token_to_predict = len(op_token_ids)

        optimzer.zero_grad()
        for _ in range(len(op_token_ids)):
            logits = model(ip_token_ids)
            # pred_token_id = torch.argmax(logits[:, -1, :], dim=-1).item()

            loss = loss_func(
                logits[:, -1, :].flatten(0, 1),
                torch.tensor(op_token_ids[0], dtype=torch.long, device=settings.device),
            )
            loss.backward()
            optimzer.step()
            ip_token_ids = torch.cat(
                [
                    ip_token_ids,
                    torch.tensor(
                        op_token_ids[0], dtype=torch.long, device=settings.device
                    ).reshape(1, 1),
                ],
                dim=1,
            )

    torch.save(model.state_dict(), "./AbbyGPT_StateDict_SFT.pt")
