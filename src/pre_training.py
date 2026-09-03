from pathlib import Path

import torch
from tokenizers import Tokenizer

from gpt import AbbyGPT
from settings import settings
from utils import count_trainable_params, create_dataset

if __name__ == "__main__":
    my_bpe_tokenizer = Tokenizer.from_file("./bpe_tokenizer.json")

    train_token_indices = torch.Tensor(
        my_bpe_tokenizer.encode(
            Path(settings.train_file).read_text(encoding="utf-8"),
        ).ids
    ).to(dtype=torch.long, device=settings.device)
    print("Train Tokens :: ", len(train_token_indices))

    val_token_indices = torch.Tensor(
        my_bpe_tokenizer.encode(
            Path(settings.val_file).read_text(encoding="utf-8"),
        ).ids
    ).to(dtype=torch.long, device=settings.device)
    xs_train, ys_train = create_dataset(train_token_indices)
    xs_val, ys_val = create_dataset(val_token_indices)
    train_num_batches = xs_train.shape[0]
    val_num_batches = xs_val.shape[0]
    print("Train Num Batches :: ", train_num_batches)

    model = AbbyGPT(vocab_size=my_bpe_tokenizer.get_vocab_size()).to(settings.device)
    print("Num Trainable Params :: ", count_trainable_params(model))
    loss_func = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    model.train()

    best_val_loss = float("inf")
    for e in range(settings.epochs):
        model.train()
        loss_for_epoch: float = 0
        for i in range(train_num_batches):
            optimizer.zero_grad()
            probs = model(xs_train[i, ...])

            loss = loss_func(
                probs.flatten(0, 1), ys_train[i, ...].flatten()
            )  # require raw logits
            loss.backward()
            optimizer.step()
            loss_for_epoch += loss.item()

        # calculate validation loss
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for i in range(val_num_batches):
                logits = model(xs_val[i, ...])
                total_val_loss += loss_func(
                    logits.flatten(0, 1), ys_val[i, ...].flatten()
                ).item()
            avg_val_loss = total_val_loss / val_num_batches
            if best_val_loss > avg_val_loss:
                best_val_loss = avg_val_loss
                torch.save(model.state_dict(), "AbbyGPT_StateDict.pt")
        print(
            f"Epoch {e + 1} Training Loss {loss_for_epoch / train_num_batches:.4f} Validation Loss {avg_val_loss:.4f}"
        )
