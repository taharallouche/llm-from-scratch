from typing import ClassVar

import datasets as ds
import torch
from torch.utils.data import WeightedRandomSampler

from llm.io.hugging_face_datasets.base import HFDataLoaderFactory, HFTorchDataset


class PaperReviewsDataset(HFTorchDataset): ...


class PaperReviewsDataLoaderFactory(HFDataLoaderFactory[PaperReviewsDataset]):
    hf_dataset_path: ClassVar[str] = "Vidushee/iclr-papers-with-code-1k"
    hf_data_files: ClassVar[str] = "all_1051_papers.jsonl"
    dataset_class: ClassVar[type[PaperReviewsDataset]] = PaperReviewsDataset
    feature_name: ClassVar[str] = "title"
    label_name: ClassVar[str] = "status"

    def _preprocess(self, raw: ds.DatasetDict, seed=42) -> ds.DatasetDict:
        assert list(raw.keys()) == ["train"]
        dataset: ds.Dataset = raw["train"]
        to_remove = [
            col for col in dataset.column_names if col not in ["title", "status"]
        ]
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

        return ds.DatasetDict(
            {"train": train_ds, "validation": validation_ds, "test": test_ds}
        )

    def _sampler(self, dataset: PaperReviewsDataset) -> WeightedRandomSampler:
        labels = torch.tensor(dataset.labels)

        class_counts = torch.bincount(labels)
        class_weights = 1.0 / class_counts.float()

        sample_weights = class_weights[labels]

        return WeightedRandomSampler(
            weights=sample_weights.tolist(),
            num_samples=len(labels),
            replacement=True,
        )
