import logging
import time
from pathlib import Path

import torch
from safetensors.torch import save_model
from tokenizers import Tokenizer
from torch.profiler import record_function
from torch.utils.data import DataLoader
from transformers import GPT2ForSequenceClassification
from transformers.modeling_outputs import SequenceClassifierOutputWithPast

from llm.device import get_device
from llm.fine_tuning.classification.training_logs import (
    FineTuningLogs,
    FineTuningLosses,
)
from llm.io.hugging_face_datasets.base import HFDataLoaderFactory
from llm.io.hugging_face_datasets.movie_reviews import MovieReviewsDataLoaderFactory
from llm.io.utils import experiments_path

LOGGER = logging.getLogger(__name__)


def setup_model() -> GPT2ForSequenceClassification:
    tokenizer = Tokenizer.from_pretrained("gpt2")
    model = GPT2ForSequenceClassification.from_pretrained(
        "openai-community/gpt2",
        num_labels=2,
    )
    model.config.pad_token_id = tokenizer.token_to_id("<|endoftext|>")

    for param in model.parameters():
        param.requires_grad = False

    for param in model.transformer.h[-1].parameters():
        param.requires_grad = True

    for param in model.transformer.ln_f.parameters():
        param.requires_grad = True

    for param in model.score.parameters():
        param.requires_grad = True

    return model


def pipeline(
    model: GPT2ForSequenceClassification,
    dataloader_factory: HFDataLoaderFactory,
    n_epochs: int,
    eval_freq: int | None = None,
    max_batches_eval: int | None = None,
    device: str | torch.device = "cpu",
    checkpoint_name: str | None = None,
    plot_losses: bool = False,
    export_path: Path | None = None,
    profile: bool = False,
    profile_steps: int = 6,
):
    # TODO: refactor common logic with gpt2 from scratch

    model.to(device)  # type: ignore[arg-type]

    dataloaders = dataloader_factory.create()
    total_batches = len(dataloaders.train)

    train_losses: list[float] = []
    validation_losses: list[float] = []

    profile_dir = export_path / "profiler" if profile and export_path else None
    device = torch.device(device)
    is_cuda = device.type == "cuda"

    if is_cuda:
        torch.cuda.synchronize(device)

    start_time = time.perf_counter()

    if is_cuda and profile_dir is not None:
        profile_dir.mkdir(parents=True, exist_ok=True)
        torch.cuda.memory._record_memory_history(enabled="all")

    try:
        for epoch in range(epochs):
            LOGGER.info("Epoch %d out of %d.", epoch + 1, n_epochs)
            model.train()

            for i, (input_ids, labels) in enumerate(dataloaders.train, start=1):
                LOGGER.info("Batch %d out of %d", i, total_batches)

                input_ids: torch.Tensor = input_ids.to(device)  # type: ignore[no-redef]
                labels: torch.Tensor = labels.to(device)  # type: ignore[no-redef]

                # attention_mask = (input_ids != pad_token_id).long()
                # no attention mask to reproduce my implementation

                optimizer.zero_grad()
                with record_function("forward_and_loss"):
                    outputs: SequenceClassifierOutputWithPast = model(
                        input_ids=input_ids,
                        # attention_mask=attention_mask,
                        labels=labels,
                    )

                    loss = outputs.loss
                    assert isinstance(loss, torch.Tensor)

                with record_function("backward"):
                    loss.backward()

                with record_function("optimizer_step"):
                    optimizer.step()

                global_step = epoch * total_batches + i

                if is_cuda and profile_dir is not None and global_step == profile_steps:
                    snapshot_path = profile_dir / "memory_snapshot.pickle"
                    torch.cuda.memory._dump_snapshot(str(snapshot_path))
                    torch.cuda.memory._record_memory_history(enabled=None)
                    LOGGER.info("CUDA memory snapshot saved to %s", snapshot_path)

                if eval_freq and global_step % eval_freq == 0:
                    train_loss, validation_loss = evaluate_model(
                        model=model,
                        device=device,
                        train_loader=dataloaders.train,
                        val_loader=dataloaders.validation,
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
        if is_cuda and profile_dir is not None:
            try:
                torch.cuda.memory._record_memory_history(enabled=None)
            except Exception:
                pass

    if is_cuda:
        torch.cuda.synchronize(device)
    elapsed_seconds = time.perf_counter() - start_time

    LOGGER.info("Fine-tuning complete!")

    losses = FineTuningLosses(
        train=train_losses,
        validation=validation_losses,
    )

    finetuning_logs = FineTuningLogs(
        checkpoint_name=checkpoint_name or "",
        n_epochs=n_epochs,
        batch_size=dataloader_factory.batch_size,
        max_batches_eval=max_batches_eval,
        eval_freq=eval_freq,
        learning_rate=learning_rate,
        losses=losses,
        training_time=elapsed_seconds,
    )

    if export_path:
        export_path.mkdir(parents=True, exist_ok=True)

        finetuning_logs.plot(path=export_path / "losses_curve.png", show=plot_losses)
        finetuning_logs.save_run_metadata(export_path)

        if checkpoint_name:
            save_model(model, str(export_path / f"{checkpoint_name}.safetensors"))

    return finetuning_logs


def compute_loss_batch(
    input_batch: torch.Tensor,
    target_batch: torch.Tensor,
    model: GPT2ForSequenceClassification,
    device: str | torch.device = "cpu",
    class_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)
    if class_weights is not None:
        class_weights = class_weights.to(device)

    output: SequenceClassifierOutputWithPast = model(input_batch)
    loss = output.loss
    assert isinstance(loss, torch.Tensor)

    return loss


def compute_loss(
    dataloader: DataLoader,
    model: GPT2ForSequenceClassification,
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
    model: GPT2ForSequenceClassification,
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


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    device = get_device()
    print("device: ", device)

    model = setup_model()

    learning_rate = 0.001
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    epochs = 3

    finetuning_logs = pipeline(
        model=model,
        n_epochs=3,
        eval_freq=500000000000,
        max_batches_eval=50,
        dataloader_factory=MovieReviewsDataLoaderFactory(
            tokenizer=Tokenizer.from_pretrained("gpt2"),
            batch_size=20,
            max_length=model.config.n_positions,
        ),
        checkpoint_name="movie_reviews_hf_gpu_no_eval",
        export_path=experiments_path() / "movie_reviews_hf_gpu_no_eval",
        device=device,
        plot_losses=True,
        profile=True,
        profile_steps=5,
    )

    print("training time in s", finetuning_logs.training_time)
