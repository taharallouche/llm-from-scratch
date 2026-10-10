import logging
from collections.abc import Callable

import torch
from torch.utils.data import DataLoader

from llm.fine_tuning.classification.classifier import (
    setup_binary_classification_fine_tuning,
)
from llm.fine_tuning.classification.evaluate import compute_loss_batch, evaluate_model
from llm.fine_tuning.pretrained import pretrained_gpt_124M
from llm.gpt import GPTModel
from llm.io.movie_reviews import create_dataloaders as movies_dataloader_factory
from llm.io.paper_reviews import create_dataloaders as paper_dataloader_factory
from llm.io.utils import checkpoint_path

LOGGER = logging.getLogger(__name__)


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

    step = 0
    for epoch in range(n_epochs):
        LOGGER.info(f"Epoch {epoch} out of {n_epochs}.")
        model.train()
        for i, (input_batch, target_batch) in enumerate(train_loader):
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

            if step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    model,
                    train_loader,
                    validation_loader,
                    device,
                    num_batches=max_batches_eval,
                )
                LOGGER.info(
                    f"Train Loss: {train_loss.item()}, Validation Loss: {val_loss.item()}."
                )

            step += 1

    LOGGER.info("Fine-tuning complete !")


def pipeline(
    dataloader_factory: Callable[..., tuple[DataLoader, DataLoader, DataLoader]],
    batch_size: int,
    n_epochs: int,
    eval_freq: int,
    max_batches_eval: int | None = None,
    checkpoint_name: str | None = None,
    device="cpu",
):
    model = setup_binary_classification_fine_tuning(
        pretrained_gpt_124M(),
        trf_blocks_layer="trf_blocks",
        final_norm_layer="final_norm",
    )
    head_params = list(model.out_head.parameters())
    trf_params = [
        p
        for n, p in model.named_parameters()
        if p.requires_grad and "out_head" not in n
    ]

    optimizer = torch.optim.AdamW(
        [
            {"params": trf_params, "lr": 5e-5},
            {"params": head_params, "lr": 1e-3},
        ],
        weight_decay=0.01,
    )

    train, val, _ = dataloader_factory(batch_size=batch_size)

    train_classifier(
        model,
        optimizer,
        train,
        val,
        n_epochs=n_epochs,
        eval_freq=eval_freq,
        max_batches_eval=max_batches_eval,
        device=device,
    )

    if checkpoint_name:
        torch.save(model.state_dict(), checkpoint_path(checkpoint_name))


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    pipeline(
        batch_size=10,
        n_epochs=5,
        eval_freq=10,
        max_batches_eval=10,
        dataloader_factory=movies_dataloader_factory,
        checkpoint_name=None,
    )
