import torch

from settings import settings


def count_trainable_params(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def create_dataset(token_indices: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
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
    if xts and yts:
        # IGNORING LAST BATCH
        # number of examples in the batch is BATCH_SIZE, last batch may short of examples so ignoring it
        # xs.append(xts)
        # ys.append(yts)
        pass

    xs = torch.Tensor(xs).to(dtype=torch.long, device=settings.device)
    ys = torch.Tensor(ys).to(dtype=torch.long, device=settings.device)
    return (xs, ys)


def divide_tiny_stories():
    from pathlib import Path

    source_text = Path(settings.source_datafile).read_text(encoding="utf-8")
    paras = source_text.split(
        "\n\n"
    )  # rudimentry way to detecting paragraphs in text file
    split_at = int(settings.train_split * len(paras))

    Path("../data/tiny_stories_train.txt").write_text(
        "\n\n".join(paras[0:split_at]),
        encoding="utf-8",
    )
    Path("../data/tiny_stories_val.txt").write_text(
        "\n\n".join(paras[split_at:]),
        encoding="utf-8",
    )


if __name__ == "__main__":
    divide_tiny_stories()
