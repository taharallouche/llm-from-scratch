import json
import logging
import socket
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from safetensors.torch import save_model
from tokenizers import Tokenizer
from torch.autograd.profiler import record_function
from torch.utils.data import DataLoader

from llm.config import GPTConfig
from llm.device import get_device
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
from llm.io.utils import experiments_path

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


TIME_FORMAT_STR: str = "%b_%d_%H_%M_%S"


def make_trace_handler(profile_dir: Path):
    profile_dir.mkdir(parents=True, exist_ok=True)

    def trace_handler(prof: torch.profiler.profile) -> None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        hostname = socket.gethostname()
        prefix = f"{hostname}_{timestamp}"

        # CPU and GPU operator trace, suitable for Chrome/Perfetto.
        trace_path = profile_dir / f"{prefix}_trace.json.gz"
        prof.export_chrome_trace(str(trace_path))

        # Memory allocation timeline, including an HTML visualization.
        if torch.cuda.is_available():
            memory_path = profile_dir / f"{prefix}_memory.html"
            prof.export_memory_timeline(
                str(memory_path),
                device="cuda:0",
            )

        LOGGER.info("Profiler trace saved to %s", trace_path)

    return trace_handler


def train_classifier(
    model: GPTModel,
    optimizer: torch.optim.Optimizer,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    n_epochs: int,
    eval_freq: int,
    max_batches_eval: int | None = None,
    device: str | torch.device = "cpu",
    profile_dir: Path | None = None,
    profile_steps: int = 6,
) -> tuple[FineTuningLosses, float]:
    model.to(device)

    total_batches = len(train_loader)

    train_losses = []
    validation_losses = []

    device = torch.device(device)
    is_cuda = device.type == "cuda"

    # Profile CUDA only when training on a CUDA device.
    activities = [torch.profiler.ProfilerActivity.CPU]

    if is_cuda:
        activities.append(torch.profiler.ProfilerActivity.CUDA)

        # Synchronize pending CUDA work before measuring elapsed time.
        torch.cuda.synchronize(device)

    start_time = time.perf_counter()

    # Profiling is optional.
    profiler = None

    if profile_dir is not None:
        profiler = torch.profiler.profile(
            activities=activities,
            schedule=torch.profiler.schedule(
                wait=2,
                warmup=5,
                active=profile_steps,
                repeat=1,
            ),
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
            on_trace_ready=make_trace_handler(profile_dir),
        )

    try:
        if profiler is None:
            # Use a null context when profiling is disabled.
            from contextlib import nullcontext

            profiler_context = nullcontext()
        else:
            profiler_context = profiler

        with profiler_context as prof:
            for epoch in range(n_epochs):
                LOGGER.info("Epoch %d out of %d.", epoch + 1, n_epochs)

                model.train()

                for i, (input_batch, target_batch) in enumerate(train_loader, start=1):
                    LOGGER.info("Batch %d out of %d", i, total_batches)

                    optimizer.zero_grad()

                    with record_function("forward_and_loss"):
                        loss = compute_loss_batch(
                            input_batch,
                            target_batch,
                            model,
                            device,
                        )

                    with record_function("backward"):
                        loss.backward()

                    with record_function("optimizer_step"):
                        optimizer.step()

                    global_step = epoch * total_batches + i

                    # Mark the completion of this training iteration.
                    if profiler is not None:
                        prof.step()

                    # Evaluation is deliberately outside the profiled
                    # training steps when using the configuration below.
                    if global_step % eval_freq == 0:
                        train_loss, validation_loss = evaluate_model(
                            model,
                            train_loader,
                            validation_loader,
                            device,
                            num_batches=max_batches_eval,
                        )

                        LOGGER.info(
                            "Train Loss: %s, Validation Loss: %s.",
                            train_loss,
                            validation_loss,
                        )

                        train_losses.append(train_loss)
                        validation_losses.append(validation_loss)

    finally:
        # Exiting the context finalizes the profiler and its trace.
        pass

    if is_cuda:
        torch.cuda.synchronize(device)

    elapsed_seconds = time.perf_counter() - start_time

    LOGGER.info("Fine-tuning complete!")

    return (
        FineTuningLosses(
            train=train_losses,
            validation=validation_losses,
        ),
        elapsed_seconds,
    )


def pipeline(
    model: GPTModel,
    dataloader_factory: HFDataLoaderFactory,
    n_epochs: int,
    eval_freq: int,
    max_batches_eval: int | None = None,
    checkpoint_name: str | None = None,
    device: str | torch.device = "cpu",
    plot_losses: bool = False,
    export_path: Path | None = None,
    profile: bool = False,
    profile_steps: int = 6,
) -> FineTuningLogs:

    learning_rate = 0.001
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    dataloader_split = dataloader_factory.create()

    profile_dir = None

    if profile:
        profile_dir = (
            export_path / "profiler" if export_path is not None else Path("profiler")
        )

    losses, training_time = train_classifier(
        model,
        optimizer,
        dataloader_split.train,
        dataloader_split.validation,
        n_epochs=n_epochs,
        eval_freq=eval_freq,
        max_batches_eval=max_batches_eval,
        device=device,
        profile_dir=profile_dir,
        profile_steps=profile_steps,
    )

    finetuning_logs = FineTuningLogs(
        checkpoint_name=checkpoint_name or "",
        n_epochs=n_epochs,
        batch_size=dataloader_factory.batch_size,
        max_batches_eval=max_batches_eval,
        eval_freq=eval_freq,
        learning_rate=learning_rate,
        losses=losses,
        training_time=training_time,
    )

    if export_path:
        export_path.mkdir(parents=True, exist_ok=True)

        finetuning_logs.plot(
            path=export_path / "losses_curve.png",
            show=plot_losses,
        )
        finetuning_logs.save_run_metadata(export_path)

        if checkpoint_name:
            save_model(
                model,
                str(export_path / f"{checkpoint_name}.safetensors"),
            )

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

    device = get_device()

    print("device: ", device)

    finetuning_logs = pipeline(
        model=model,
        n_epochs=3,
        eval_freq=5000000000000,
        max_batches_eval=50,
        dataloader_factory=MovieReviewsDataLoaderFactory(
            tokenizer=Tokenizer.from_pretrained("gpt2"),
            batch_size=20,
            max_length=GPTConfig().context_length,
        ),
        checkpoint_name="movie_reviews_gpu",
        export_path=experiments_path() / "movie_reviews_gpu",
        device=device,
        profile=True,
        profile_steps=15,
    )

    print("training time in s", finetuning_logs.training_time)
