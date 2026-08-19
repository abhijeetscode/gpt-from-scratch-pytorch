from typing import cast

import torch

from gpt import AbbyGPT
from my_tokenizer import MyCharTokenizer
from settings import settings

if __name__ == "__main__":
    with open("../data/verdict.txt") as fp:
        data = fp.read()

    tokenizer = MyCharTokenizer()
    tokenizer.fit(data)
    token_indices: torch.Tensor = cast(
        torch.Tensor, tokenizer.encode(data, return_pt=True)
    )
    decoded_data = tokenizer.decode(token_indices.tolist())
    assert decoded_data == data, "something is wrong with tokenizer"

    # assert torch.max(token_indices).item()+1 == tokenizer.vocab_size, "vocab size and max token index not matching"
    print("Total number of tokens :: ", len(token_indices))

    xs = []
    ys = []
    for i in range(
        0, token_indices.shape[0] - settings.context_length, settings.context_length
    ):
        start_index, end_index = i, i + settings.context_length
        x_train = token_indices[start_index:end_index]

        y_train = token_indices[start_index + 1 : end_index + 1]
        xs.append(x_train)
        ys.append(y_train)

    xs = torch.vstack(xs).to(settings.device)
    ys = torch.vstack(ys).to(settings.device)

    model = AbbyGPT(vocab_size=tokenizer.vocab_size).to(settings.device)
    print(type(model))
    loss_func = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    model.train()
    for e in range(settings.epochs):
        optimizer.zero_grad()
        probs = model(xs)
        probs_flatten = probs.flatten(0, 1)
        ys_flatten = ys.flatten()
        loss = loss_func(probs_flatten, ys_flatten)  # require raw logits
        loss.backward()
        optimizer.step()
        print("Loss :: ", loss.item())

    model.eval()
    xs_test = cast(torch.Tensor, tokenizer.encode("I HAD al", return_pt=True))
    xs_test = xs_test.to(settings.device)
    prob_test = model(xs_test)
    pred_test = torch.argmax(prob_test, dim=-1)
    op: str = tokenizer.decode(pred_test.flatten().tolist())
    print("===> ", op)
