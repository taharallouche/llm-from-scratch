from pathlib import Path


def data_path() -> Path:
    return Path(__file__).parents[3] / "data"


def load_data(dataset: str) -> str:
    dataset_file = data_path() / "input" / f"{dataset}.txt"
    return dataset_file.read_text()


def checkpoint_path(name: str) -> Path:
    return data_path() / "artefacts" / name
