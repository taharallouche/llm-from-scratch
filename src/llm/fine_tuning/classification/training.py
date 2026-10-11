import json
import logging
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import tiktoken
import torch
from torch.utils.data import DataLoader

from llm.config import GPTConfig
from llm.fine_tuning.classification.classifier import (
    setup_binary_classification_fine_tuning,
)
from llm.fine_tuning.classification.evaluate import compute_loss_batch, evaluate_model
from llm.fine_tuning.pretrained import pretrained_gpt_124M
from llm.gpt import GPTModel
from llm.io.hugging_face_datasets.base import HFDataLoaderFactory
from llm.io.hugging_face_datasets.movie_reviews import (
    MovieReviewsDataLoaderFactory,
)
from llm.io.utils import data_path

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class FineTuningLosses:
    train: list[float]
    validation: list[float]


@dataclass(frozen=True)
class FineTuningLogs:
    n_epochs: int
    batch_size: int
    max_batches_eval: int | None
    eval_freq: int
    learning_rate: float
    checkpoint_name: str

    losses: FineTuningLosses

    def save_run_metadata(self, folder: Path) -> None:
        file_path = folder / "run_metdata.json"
        file_path.touch()

        with open(file_path, mode="w+") as f:
            json.dump(
                {
                    "checkpoint_name": self.checkpoint_name,
                    "n_epochs": self.n_epochs,
                    "batch_size": self.batch_size,
                    "max_batches_eval": self.max_batches_eval,
                    "eval_freq": self.eval_freq,
                    "learning_rate": self.learning_rate,
                },
                f,
            )

    def plot(self, path: Path | None = None, show: bool = False) -> None:
        steps = [(i + 1) * self.eval_freq for i in range(len(self.losses.train))]

        plt.figure(figsize=(8, 5))
        plt.plot(
            steps, self.losses.train, label="Train Loss", marker="o", linestyle="-"
        )
        plt.plot(
            steps,
            self.losses.validation,
            label="Validation Loss",
            marker="x",
            linestyle="--",
        )

        plt.xlabel("Batch / Step")
        plt.ylabel("Loss")

        if self.checkpoint_name != "":
            plt.title(f"Fine-Tuning Loss - Checkpoint: {self.checkpoint_name}")
        else:
            plt.title("Fine-Tuning Loss")

        plt.legend()
        plt.grid(True, linestyle=":", alpha=0.7)
        plt.tight_layout()

        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(path, dpi=300, bbox_inches="tight")

        if show:
            plt.show()

        plt.close()


def train_classifier(
    model: GPTModel,
    optimizer: torch.optim.Optimizer,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    n_epochs: int,
    eval_freq: int,
    max_batches_eval: int | None = None,
    device="cpu",
):

    total_batches = len(train_loader)

    train_losses = []
    validation_losses = []

    for epoch in range(n_epochs):
        LOGGER.info(f"Epoch {epoch} out of {n_epochs}.")
        for i, (input_batch, target_batch) in enumerate(train_loader, start=1):
            model.train()
            LOGGER.info(f"Batch {i} out of {total_batches}")
            optimizer.zero_grad()
            loss = compute_loss_batch(
                input_batch,
                target_batch,
                model,
                device,
            )

            loss.backward()

            optimizer.step()

            if (epoch * total_batches + i) % eval_freq == 0:
                train_loss, validation_loss = evaluate_model(
                    model,
                    train_loader,
                    validation_loader,
                    device,
                    num_batches=max_batches_eval,
                )
                LOGGER.info(
                    f"Train Loss: {train_loss.item()}, Validation Loss: {validation_loss.item()}."
                )

                train_losses.append(train_loss)
                validation_losses.append(validation_loss)

    LOGGER.info("Fine-tuning complete !")

    return FineTuningLosses(train=train_losses, validation=validation_losses)


def pipeline(
    model: GPTModel,
    dataloader_factory: HFDataLoaderFactory,
    n_epochs: int,
    eval_freq: int,
    max_batches_eval: int | None = None,
    checkpoint_name: str | None = None,
    device="cpu",
    plot_losses: bool = False,
    export_path: Path | None = None,
) -> FineTuningLogs:

    learning_rate = 0.001
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    dataloader_split = dataloader_factory.create()

    losses = train_classifier(
        model,
        optimizer,
        dataloader_split.train,
        dataloader_split.validation,
        n_epochs=n_epochs,
        eval_freq=eval_freq,
        max_batches_eval=max_batches_eval,
        device=device,
    )

    finetuning_logs = FineTuningLogs(
        checkpoint_name=checkpoint_name or "",
        n_epochs=n_epochs,
        batch_size=dataloader_factory.batch_size,
        max_batches_eval=max_batches_eval,
        eval_freq=eval_freq,
        learning_rate=learning_rate,
        losses=losses,
    )

    if export_path:
        export_path.mkdir(parents=True, exist_ok=True)
        finetuning_logs.plot(path=export_path / "losses_curve", show=plot_losses)
        finetuning_logs.save_run_metadata(export_path)

        if checkpoint_name:
            torch.save(model.state_dict(), export_path / f"{checkpoint_name}.pth")

    return finetuning_logs


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    model = setup_binary_classification_fine_tuning(
        pretrained_gpt_124M(),
        trf_blocks_layer="trf_blocks",
        final_norm_layer="final_norm",
    )

    finetuning_logs = pipeline(
        model=model,
        n_epochs=3,
        eval_freq=50,
        max_batches_eval=50,
        dataloader_factory=MovieReviewsDataLoaderFactory(
            tokenizer=tiktoken.get_encoding("gpt2"),
            batch_size=20,
            max_length=GPTConfig().context_length,
        ),
        checkpoint_name="movie_reviews",
        export_path=data_path() / "experiments" / "movie_reviews",
    )
