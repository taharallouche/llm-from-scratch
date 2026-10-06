from functools import partial

import datasets as ds
import tiktoken
import torch
from torch.utils.data import DataLoader, Dataset


def load_data() -> ds.DatasetDict:
    return ds.load_dataset(
        "Vidushee/iclr-papers-with-code-1k", data_files="all_1051_papers.jsonl"
    )


def pre_process_data(
    dataset: ds.DatasetDict, seed: int = 42
) -> tuple[ds.Dataset, ds.Dataset, ds.Dataset]:
    assert list(dataset.keys()) == ["train"]
    dataset: ds.Dataset = dataset["train"]
    to_remove = [col for col in dataset.column_names if col not in ["title", "status"]]
    dataset = dataset.remove_columns(to_remove)

    def update_accepted_label(row: dict[str, str]) -> dict[str, str]:
        if row["status"] == "accepted_poster":
            row["status"] = "accepted"
        return row

    dataset = dataset.map(update_accepted_label)

    dataset = dataset.cast_column(
        "status", feature=ds.ClassLabel(names=["rejected", "accepted"])
    )
    split = dataset.train_test_split(
        train_size=0.8, stratify_by_column="status", seed=seed
    )
    train_ds = split["train"]

    validation_test_split = split["test"].train_test_split(
        test_size=0.5, stratify_by_column="status", seed=seed
    )

    validation_ds, test_ds = (
        validation_test_split["train"],
        validation_test_split["test"],
    )

    return train_ds, validation_ds, test_ds


class PaperReviews(Dataset):
    def __init__(
        self,
        data: ds.Dataset,
        tokenizer: tiktoken.Encoding,
        max_length: int | None = None,
        pad_token_id: int = 50256,
    ) -> None:
        self.data = data

        encoded_titles = [
            tokenizer.encode(paper["title"][0]) for paper in data.iter(batch_size=1)
        ]

        if max_length:
            self.max_length = max_length
        else:
            self.max_length = max(len(encoded) for encoded in encoded_titles)

        self.encoded_titles = [
            title + [pad_token_id] * (self.max_length - len(title))
            for title in encoded_titles
        ]

        self.labels = [paper["status"] for paper in data.iter(batch_size=1)]

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.tensor(self.encoded_titles[index], dtype=torch.long), torch.tensor(
            self.labels[index], dtype=torch.int8
        )

    def __len__(self) -> int:
        return len(self.labels)


def create_dataloaders(
    batch_size: int = 8, num_worker: int = 0, seed: int = 42
) -> tuple[DataLoader, DataLoader, DataLoader]:
    train_ds, validation_ds, test_ds = pre_process_data(load_data(), seed=seed)

    paper_reviews = partial(PaperReviews, tokenizer=tiktoken.get_encoding("gpt2"))

    train, validation, test = (
        paper_reviews(train_ds),
        paper_reviews(validation_ds),
        paper_reviews(test_ds),
    )

    train_loader = DataLoader(train, batch_size=batch_size, num_workers=num_worker)
    validation_loader = DataLoader(
        validation, batch_size=batch_size, num_workers=num_worker
    )
    test_loader = DataLoader(test, batch_size=batch_size, num_workers=num_worker)

    return train_loader, validation_loader, test_loader
