from dataclasses import dataclass
from functools import partial

import numpy as np
import torch
from numpy.typing import NDArray
from safetensors.torch import load_model
from sklearn.metrics import classification_report, confusion_matrix
from tokenizers import Tokenizer
from torch.utils.data import DataLoader
from transformers import GPT2ForSequenceClassification
from transformers.modeling_outputs import SequenceClassifierOutputWithPast

from llm.fine_tuning.classification.hugging_face import setup_model
from llm.gpt import GPTModel
from llm.io.hugging_face_datasets.movie_reviews import MovieReviewsDataLoaderFactory
from llm.io.utils import experiments_path


def compute_loss_batch(
    input_batch: torch.Tensor,
    target_batch: torch.Tensor,
    model: GPTModel,
    device: str | torch.device = "cpu",
    class_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)
    if class_weights is not None:
        class_weights = class_weights.to(device)

    logits = model(input_batch)  # n_batch x seq_length x 2

    last_logits = logits[:, -1, :]  # n_batch x 2

    return torch.nn.functional.cross_entropy(
        last_logits, target_batch, weight=class_weights
    )


def compute_loss(
    dataloader: DataLoader,
    model: GPTModel,
    max_batches: int | None = None,
    device: str | torch.device = "cpu",
    class_weights: torch.Tensor | None = None,
) -> float:

    total_losses = 0.0

    for batches_seen, (input_batch, target_batch) in enumerate(dataloader, start=1):
        batch_loss = compute_loss_batch(
            input_batch, target_batch, model, device, class_weights=class_weights
        )

        total_losses += batch_loss.item()

        if max_batches and batches_seen == max_batches:
            break

    return total_losses / batches_seen


def evaluate_model(
    model: GPTModel,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device,
    num_batches: int | None,
) -> tuple[float, float]:
    model.eval()
    with torch.no_grad():
        train_loss = compute_loss(
            train_loader, model, device=device, max_batches=num_batches
        )
        val_loss = compute_loss(
            val_loader, model, device=device, max_batches=num_batches
        )
    model.train()
    return train_loss, val_loss


def get_predictions_and_targets(
    dataloader: DataLoader,
    model: GPTModel | GPT2ForSequenceClassification,
    device: str | torch.device = "cpu",
    num_batches=None,
) -> tuple[torch.Tensor, torch.Tensor]:
    model.eval()

    predictions = []
    targets = []

    with torch.no_grad():
        for i, (input_batch, target_batch) in enumerate(dataloader):
            if isinstance(model, GPTModel):
                logits = model(input_batch.to(device))[:, -1]
            elif isinstance(model, GPT2ForSequenceClassification):
                output: SequenceClassifierOutputWithPast = model(input_batch.to(device))
                logits = output.logits
            predictions.extend(logits.argmax(dim=-1).cpu())
            targets.extend(target_batch.reshape(-1).cpu())

            if num_batches is not None and i + 1 >= num_batches:
                break

    return torch.tensor(predictions), torch.tensor(targets)


@dataclass(frozen=True)
class DatasetEvaluation:
    classification_report: str
    confusion_matrix: NDArray

    def __str__(self) -> str:
        return (
            "Classification Report\n"
            "=====================\n"
            f"{self.classification_report}\n"
            "Confusion Matrix\n"
            "================\n"
            f"{np.array2string(self.confusion_matrix)}"
        )

    @classmethod
    def compute(
        cls,
        model: GPTModel,
        dataloader: DataLoader,
        num_batches: int | None = None,
        device: str | torch.device = "cpu",
    ) -> "DatasetEvaluation":
        predictions, targets = get_predictions_and_targets(
            dataloader=dataloader, model=model, num_batches=num_batches, device=device
        )
        return cls(
            classification_report=classification_report(targets, predictions),
            confusion_matrix=confusion_matrix(targets, predictions),
        )


@dataclass(frozen=True)
class EvaluationReport:
    train: DatasetEvaluation
    validation: DatasetEvaluation
    test: DatasetEvaluation

    @classmethod
    def generate(
        cls,
        model,
        train_loader,
        validation_loader,
        test_loader,
        num_batches: int | None = None,
        device: str | torch.device = "cpu",
    ) -> "EvaluationReport":

        eval_compute = partial(
            DatasetEvaluation.compute,
            model=model,
            num_batches=num_batches,
            device=device,
        )

        return cls(
            train=eval_compute(dataloader=train_loader),
            validation=eval_compute(dataloader=validation_loader),
            test=eval_compute(dataloader=test_loader),
        )

    def __str__(self) -> str:
        return "\n\n".join(
            (
                f"TRAIN SET\n{'=' * 60}\n{self.train}",
                f"VALIDATION SET\n{'=' * 60}\n{self.validation}",
                f"TEST SET\n{'=' * 60}\n{self.test}",
            )
        )


if __name__ == "__main__":
    # model = setup_binary_classification_model(
    #     model=GPTModel(GPTConfig(drop_rate=0.0, qkv_bias=True)),
    # )

    model = setup_model()

    load_model(
        model,
        experiments_path()
        / "movie_reviews_hf_gpu_no_eval"
        / "movie_reviews_hf_gpu_no_eval.safetensors",
    )

    loaders = MovieReviewsDataLoaderFactory(
        tokenizer=Tokenizer.from_pretrained("gpt2"), batch_size=20
    ).create()

    report = EvaluationReport.generate(
        model,
        train_loader=loaders.train,
        validation_loader=loaders.validation,
        test_loader=loaders.test,
    )

    print(report)
