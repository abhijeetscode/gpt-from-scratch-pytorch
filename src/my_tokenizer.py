import torch


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
