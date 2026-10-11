from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import partial
from typing import ClassVar, Literal, assert_never

import datasets as ds
import tiktoken
import torch
from torch.utils.data import DataLoader, Sampler
from torch.utils.data import Dataset as TorchDataset


@dataclass(frozen=True, kw_only=True)
class SplitDataLoader:
    train: DataLoader
    validation: DataLoader
    test: DataLoader


@dataclass(frozen=True, kw_only=True)
class SplitDataset:
    train: ds.Dataset
    validation: ds.Dataset
    test: ds.Dataset


class HFTorchDataset(TorchDataset):
    def __init__(
        self,
        data: ds.Dataset,
        tokenizer: tiktoken.Encoding,
        feature_name: str,
        label_name: str,
        max_length: int = 1024,
        pad_token_id: int = 50256,
        padding: Literal["left", "right"] = "left",
    ) -> None:
        self.data = data

        encoded_features = [
            tokenizer.encode(sample[feature_name][0])
            for sample in data.iter(batch_size=1)
        ]

        longest_sequence = max(len(encoded) for encoded in encoded_features)
        self.max_length = min(max_length, longest_sequence)

        # TODO: pad in a collate_fn in the dataloader. Takes the longest seq in the batch
        match padding:
            case "right":
                encoded_features = [
                    feature[: self.max_length] for feature in encoded_features
                ]
                self.encoded_features = [
                    feature + [pad_token_id] * (self.max_length - len(feature))
                    for feature in encoded_features
                ]
            case "left":
                encoded_features = [
                    feature[-self.max_length :] for feature in encoded_features
                ]
                self.encoded_features = [
                    [pad_token_id] * (self.max_length - len(feature)) + feature
                    for feature in encoded_features
                ]

            case _:
                assert_never(padding)

        self.labels = [sample[label_name][0] for sample in data.iter(batch_size=1)]

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.tensor(
            self.encoded_features[index], dtype=torch.long
        ), torch.tensor(self.labels[index], dtype=torch.long)

    def __len__(self) -> int:
        return len(self.labels)


class HFDataLoaderFactory:
    hf_dataset_path: ClassVar[str]
    hf_data_files: ClassVar[
        str | Sequence[str] | Mapping[str, str | Sequence[str]] | None
    ]
    dataset_class: ClassVar[type[HFTorchDataset]]
    feature_name: ClassVar[str]
    label_name: ClassVar[str]

    tokenizer: tiktoken.Encoding
    max_length: int | None = None
    pad_token_id: int = 50256
    padding: Literal["left", "right"] = "left"
    seed: int = 42

    batch_size: int
    num_workers: int = 0

    def __init__(
        self,
        tokenizer: tiktoken.Encoding,
        batch_size: int,
        max_length: int | None = None,
        pad_token_id: int = 50256,
        padding: Literal["left", "right"] = "left",
        seed: int = 42,
        num_workers: int = 0,
    ) -> None:
        self.tokenizer = tokenizer
        self.batch_size = batch_size  # TODO: pass it as argument to create
        self.max_length = max_length
        self.pad_token_id = pad_token_id
        self.padding = padding
        self.seed = seed
        self.num_workers = num_workers

    def create(self) -> SplitDataLoader:
        raw = ds.load_dataset(self.hf_dataset_path, data_files=self.hf_data_files)

        pre_processed = self._preprocess(raw, seed=self.seed)

        create_dataclass = partial(
            self.dataset_class,
            tokenizer=self.tokenizer,
            feature_name=self.feature_name,
            label_name=self.label_name,
            max_length=self.max_length,
            pad_token_id=self.pad_token_id,
            padding=self.padding,
        )

        train_ds, validation_ds, test_ds = (
            create_dataclass(pre_processed["train"]),
            create_dataclass(pre_processed["validation"]),
            create_dataclass(pre_processed["test"]),
        )

        create_dataloader = partial(
            DataLoader, batch_size=self.batch_size, num_workers=self.num_workers
        )

        sampler = self._sampler(train_ds)
        shuffle = not sampler

        return SplitDataLoader(
            train=create_dataloader(
                dataset=train_ds, sampler=self._sampler(train_ds), shuffle=shuffle
            ),
            validation=create_dataloader(dataset=validation_ds),
            test=create_dataloader(dataset=test_ds),
        )

    def _preprocess(self, raw: ds.DatasetDict, seed: int = 42) -> ds.DatasetDict:
        """
        To be overwritten by child classes if custom preprocessing is needed.
        Returned DatasetDict must have "train", "validation" and "test" keys.
        # TODO: Add validation for constraint
        """
        return raw

    def _sampler(self, dataset: TorchDataset) -> Sampler | None:
        """
        To be overwritten by child classes if custom sampling is needed.
        """
        return None
