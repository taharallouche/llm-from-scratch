import torch
from tiktoken import Encoding, get_encoding
from torch.utils.data import DataLoader, Dataset


class GPTDatasetVA(Dataset):
    def __init__(
        self, txt: str, tokenizer: Encoding, max_length: int, stride: int
    ) -> None:
        self.input_ids = []
        self.target_ids = []

        token_ids = tokenizer.encode(txt)

        for i in range(0, len(token_ids) - max_length, stride):
            self.input_ids.append(torch.tensor(token_ids[i : i + max_length]))
            self.target_ids.append(torch.tensor(token_ids[i + 1 : i + max_length + 1]))

    def __len__(self) -> int:
        return len(self.input_ids)

    def __getitem__(self, index: int) -> tuple[list[int], list[int]]:
        return self.input_ids[index], self.target_ids[index]


def create_dataloader(
    txt: str,
    batch_size: int = 4,
    max_length: int = 256,
    stride: int = 128,
    shuffle: bool = True,
    drop_last=False,
    num_workers: int = 0,
) -> DataLoader:
    tokenizer = get_encoding("gpt2")

    dataset = GPTDatasetVA(
        txt=txt, tokenizer=tokenizer, stride=stride, max_length=max_length
    )

    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
    )
