import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt


@dataclass(frozen=True)
class FineTuningLosses:
    train: list[float]
    validation: list[float]


@dataclass(frozen=True)
class FineTuningLogs:
    n_epochs: int
    batch_size: int
    max_batches_eval: int | None
    eval_freq: int | None
    learning_rate: float
    checkpoint_name: str
    training_time: float

    losses: FineTuningLosses

    def save_run_metadata(self, folder: Path) -> None:
        file_path = folder / "run_metadata.json"
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
                    "training_time": self.training_time,
                },
                f,
            )

    def plot(self, path: Path | None = None, show: bool = False) -> None:
        if self.eval_freq is None:
            return

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
