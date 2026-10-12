import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from llm.gpt import GPTModel


class DummyDataset(Dataset):
    def __init__(self, sequence_length: int = 5, n_samples: int = 10) -> None:
        self.sequences = [list(range(1, sequence_length + 1)) for _ in range(n_samples)]
        self.labels = [1] * n_samples

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.tensor(self.sequences[index], dtype=torch.long), torch.tensor(
            self.labels[index], dtype=torch.long
        )

    def __len__(self) -> int:
        return len(self.sequences)


@pytest.fixture
def dataset() -> DummyDataset:
    return DummyDataset()


@pytest.fixture
def dataloader(dataset: DummyDataset) -> DataLoader:
    return DataLoader(dataset=dataset, batch_size=5, shuffle=True, num_workers=0)


def test_train_classifier_outputs(
    small_model: GPTModel, dataloader: DataLoader
) -> None:
    # Given
    from llm.fine_tuning.classification.training import train_classifier

    n_epochs = 2
    eval_freq = 10

    expected_n_losses = (len(dataloader) * n_epochs) // eval_freq

    # When
    losses, training_time = train_classifier(
        model=small_model,
        optimizer=torch.optim.SGD(params=small_model.parameters()),
        train_loader=dataloader,
        validation_loader=dataloader,
        n_epochs=n_epochs,
        eval_freq=eval_freq,
    )

    # Then
    assert len(losses.train) == len(losses.validation) == expected_n_losses
    assert training_time > 0.0
