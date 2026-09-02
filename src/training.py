import torch
from tokenizers import Tokenizer

from gpt import AbbyGPT
from settings import settings

if __name__ == "__main__":
    with open("../data/verdict.txt") as fp:
        data = fp.read()

    my_bpe_tokenizer = Tokenizer.from_file("./bpe_tokenizer.json")
    encoding = my_bpe_tokenizer.encode(data)
    token_indices: torch.Tensor = torch.Tensor(encoding.ids).to(
        dtype=torch.long,
        device=settings.device,
    )

    # assert torch.max(token_indices).item()+1 == tokenizer.vocab_size, "vocab size and max token index not matching"
    print("Total number of tokens :: ", len(token_indices))
    print("Vocab Size :: ", my_bpe_tokenizer.get_vocab_size())

    xts = []
    yts = []

    xs = []
    ys = []

    for i in range(
        0, token_indices.shape[0] - settings.context_length, settings.context_length
    ):
        start_index, end_index = i, i + settings.context_length
        x_train = token_indices[start_index:end_index].tolist()

        y_train = token_indices[start_index + 1 : end_index + 1].tolist()
        xts.append(x_train)
        yts.append(y_train)
        if len(xts) >= settings.batch_size and len(yts) >= settings.batch_size:
            xs.append(xts)
            ys.append(yts)
            xts = []
            yts = []
    xs = torch.Tensor(xs).to(dtype=torch.long, device=settings.device)
    ys = torch.Tensor(ys).to(dtype=torch.long, device=settings.device)

    num_batches: int = xs.shape[0]
    print("== Num Batches == ", num_batches)

    model = AbbyGPT(vocab_size=my_bpe_tokenizer.get_vocab_size()).to(settings.device)
    loss_func = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    model.train()

    for e in range(settings.epochs):
        loss_for_epoch: float = 0
        for i in range(num_batches):
            optimizer.zero_grad()
            probs = model(xs[i, ...].unsqueeze(0))

            loss = loss_func(
                probs.flatten(0, 1), ys[i, ...].flatten()
            )  # require raw logits
            loss.backward()
            optimizer.step()
            loss_for_epoch += loss.item()
        print(f"Epoch {e + 1} Loss {loss_for_epoch / num_batches:.4f}")

    example_input = xs[0, 0:5, :][None, ...]

    exported_program = torch.export.export(
        model,
        args=(example_input,),
        dynamic_shapes={
            "x": {
                1: torch.export.Dim("batch_size", min=1),
            }
        },
    )
    torch.export.save(exported_program, "AbbyGPT.pt2")
