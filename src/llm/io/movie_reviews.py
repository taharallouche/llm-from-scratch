from functools import partial

import datasets as ds
import tiktoken
import torch


# TODO: refactor hf dataset loading into a base class
def load_data() -> ds.DatasetDict:
    return ds.load_dataset("cornell-movie-review-data/rotten_tomatoes")


def pre_process_data(
    dataset_dict: ds.DatasetDict, seed: int = 42
) -> tuple[ds.Dataset, ds.Dataset, ds.Dataset]:

    return dataset_dict["train"], dataset_dict["validation"], dataset_dict["test"]


class MovieReviews(torch.utils.data.Dataset):
    def __init__(
        self,
        data: ds.Dataset,
        tokenizer: tiktoken.Encoding,
        max_length: int | None = None,
        pad_token_id: int = 50256,
    ) -> None:
        self.data = data

        encoded_titles = [
            tokenizer.encode(paper["text"][0]) for paper in data.iter(batch_size=1)
        ]

        if max_length:
            self.max_length = max_length
        else:
            self.max_length = max(len(encoded) for encoded in encoded_titles)

        self.encoded_titles = [
            [pad_token_id] * (self.max_length - len(title)) + title
            for title in encoded_titles
        ]

        self.labels = [paper["label"][0] for paper in data.iter(batch_size=1)]

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.tensor(self.encoded_titles[index], dtype=torch.long), torch.tensor(
            self.labels[index], dtype=torch.long
        )

    def __len__(self) -> int:
        return len(self.labels)


def create_dataloaders(
    batch_size: int = 8, num_worker: int = 0, seed: int = 42
) -> tuple[
    torch.utils.data.DataLoader,
    torch.utils.data.DataLoader,
    torch.utils.data.DataLoader,
]:
    train_ds, validation_ds, test_ds = pre_process_data(load_data(), seed=seed)

    movie_reviews = partial(MovieReviews, tokenizer=tiktoken.get_encoding("gpt2"))

    train, validation, test = (
        movie_reviews(train_ds),
        movie_reviews(validation_ds),
        movie_reviews(test_ds),
    )

    train_loader = torch.utils.data.DataLoader(
        train, batch_size=batch_size, num_workers=num_worker, shuffle=True
    )
    validation_loader = torch.utils.data.DataLoader(
        validation, batch_size=batch_size, num_workers=num_worker
    )
    test_loader = torch.utils.data.DataLoader(
        test, batch_size=batch_size, num_workers=num_worker
    )

    return train_loader, validation_loader, test_loader
