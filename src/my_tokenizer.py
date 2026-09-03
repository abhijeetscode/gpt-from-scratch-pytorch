import torch

from settings import settings


def train_mybpe_tokenizer():
    from tokenizers import (
        Tokenizer,
        decoders,
        models,
        normalizers,
        pre_tokenizers,
        trainers,
    )

    tokenizer = Tokenizer(models.BPE(byte_fallback=True))
    tokenizer.normalizer = normalizers.NFKC()
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(
        add_prefix_space=True, use_regex=False
    )
    tokenizer.decoder = decoders.ByteLevel()
    my_trainer = trainers.BpeTrainer(
        min_frequency=2,
        special_tokens=["<EOS>"],
        show_progress=True,
        vocab_size=1000,
    )
    tokenizer.train(
        files=[str(settings.train_file)],
        trainer=my_trainer,
    )
    tokenizer.save("./bpe_tokenizer.json")

    return tokenizer


class MyCharTokenizer:
    def fit(self, x: str) -> None:
        self.char_to_int: dict[str, int] = {}
        self.int_to_char: dict[int, str] = {}
        index = 0
        for char_unit in x:
            if self.char_to_int.get(char_unit, None) is None:
                self.char_to_int[char_unit] = index
                self.int_to_char[index] = char_unit
                index += 1

        self.char_to_int["<unk>"] = index
        self.int_to_char[index] = "<unk>"
        self.vocab_size = index + 1

    def encode(self, x: str, *, return_pt=False) -> list[int] | torch.Tensor:
        unk_index = self.char_to_int["<unk>"]
        indices = [self.char_to_int.get(char_unit, unk_index) for char_unit in x]
        if return_pt:
            return torch.tensor(indices, dtype=torch.long)
        return indices

    def decode(self, x: list[int]) -> str:
        return "".join([self.int_to_char[index] for index in x])

    def save(self) -> bool:
        import json

        with open("./tokenizer.json", "w") as fp:
            json.dump(self.char_to_int, fp, indent=4)
            return True
        return False

    @classmethod
    def load(cls) -> MyCharTokenizer:
        import json

        self = cls()
        with open("./tokenizer.json") as fp:
            self.char_to_int = json.load(fp)
            self.int_to_char = {i: c for c, i in self.char_to_int.items()}
            self.vocab_size = len(self.int_to_char.keys())
            return self


if __name__ == "__main__":
    train_mybpe_tokenizer()
