from typing import ClassVar

from llm.io.hugging_face_datasets.base import HFDataLoaderFactory, HFTorchDataset


class MovieReviews(HFTorchDataset): ...


class MovieReviewsDataLoaderFactory(HFDataLoaderFactory[MovieReviews]):
    hf_dataset_path: ClassVar[str] = "cornell-movie-review-data/rotten_tomatoes"
    hf_data_files: ClassVar[None] = None
    dataset_class: ClassVar[type[MovieReviews]] = MovieReviews
    feature_name: ClassVar[str] = "text"
    label_name: ClassVar[str] = "label"
