import logging
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import tiktoken
import torch
from torch.utils.data import DataLoader

from llm.config import GPTConfig
from llm.gpt import GPTModel
from llm.inference.generate import generate
from llm.io.gpt_dataloader import create_dataloader
from llm.io.utils import checkpoint_path, load_data
from llm.training.evaluate import compute_batch_loss, evaluate_model

LOGGER = logging.getLogger(__name__)

SMALL_CONTEXT_CONFIG = GPTConfig(context_length=256)


def text_to_token_ids(text: str, tokenizer: tiktoken.Encoding) -> torch.Tensor:
    token_ids = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
    # we need to specify the dtype otherwise it's torch floats and the embedding layer raises
    return torch.tensor(token_ids, dtype=torch.long).unsqueeze(0)


def token_ids_to_text(
    token_ids: torch.Tensor, tokenizer: tiktoken.Encoding
) -> list[str]:
    return tokenizer.decode(token_ids.squeeze(0).tolist())


def train_model(
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    model: GPTModel,
    optimizer: torch.optim.Optimizer,
    n_epochs: int,
    eval_freq: int,
    eval_iter: int | None,
    device,
) -> tuple[list[float], list[float]]:
    global_step = -1

    train_losses: list[float] = []
    val_losses: list[float] = []

    for epoch in range(n_epochs):
        for input_ids, target_ids in train_dataloader:
            model.train()

            batch_loss = compute_batch_loss(input_ids, target_ids, model, device=device)
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()

            global_step += 1

            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    model, train_dataloader, val_dataloader, eval_iter, device
                )
                train_losses.append(train_loss)
                val_losses.append(val_loss)

                LOGGER.info(
                    f"Ep {epoch + 1} (Step {global_step:06d}): "
                    f"Train loss {train_loss:.3f}, "
                    f"Val loss {val_loss:.3f}"
                )

    return train_losses, val_losses


@dataclass(frozen=True)
class TrainingOutput:
    train_loss: list[float]
    validation_loss: list[float]


def pipeline(
    text_data: str,
    tokenizer: tiktoken.Encoding,
    train_ratio: float,
    batch_size: int,
    model: GPTModel,
    optimizer: torch.optim.Optimizer,
    n_epochs: int,
    eval_freq: int,
    eval_iter: int | None = None,
    checkpoint_path: Path | None = None,
    output_checkpoint_path: Path | None = None,
    seed: int = 42,
    device="cpu",
) -> None:

    num_chars = len(text_data)
    split_idx = int(train_ratio * num_chars)
    train_data, val_data = text_data[:split_idx], text_data[split_idx:]
    dataloader = partial(
        create_dataloader,
        tokenizer=tokenizer,
        batch_size=batch_size,
        max_length=model.config.context_length,
        stride=model.config.context_length,
        num_workers=0,
    )
    train_dataloader = dataloader(
        txt=train_data,
        drop_last=True,
        shuffle=True,
    )
    val_dataloader = dataloader(txt=val_data, drop_last=False, shuffle=False)

    torch.manual_seed(seed)

    if checkpoint_path is None:
        LOGGER.info("Starting training from scratch")

    else:
        LOGGER.info("Starting training from latest checkpoint")

        checkpoints = torch.load(checkpoint_path)
        model.load_state_dict(checkpoints["model"], strict=True)
        optimizer.load_state_dict(checkpoints["optimizer"])

        LOGGER.info("State dicts loaded")

    train_losses, val_losses = train_model(
        train_dataloader,
        val_dataloader,
        model=model,
        optimizer=optimizer,
        n_epochs=n_epochs,
        eval_freq=eval_freq,
        eval_iter=eval_iter,
        device=device,
    )

    LOGGER.info("Training complete.")

    if output_checkpoint_path:
        LOGGER.info("Exporting checkpoint")

        torch.save(
            {"model": model.state_dict(), "optimizer": optimizer.state_dict()},
            output_checkpoint_path,
        )

    return TrainingOutput(train_loss=train_losses, validation_loss=val_losses)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    tokenizer = tiktoken.get_encoding("gpt2")

    model = GPTModel(SMALL_CONTEXT_CONFIG)
    optimizer = torch.optim.AdamW(params=model.parameters())

    text_data = load_data("the-verdict")

    checkpoint_file_path = checkpoint_path("model_and_optimizer.pth")

    pipeline(
        text_data=text_data,
        tokenizer=tokenizer,
        train_ratio=0.9,
        batch_size=2,
        model=model,
        optimizer=optimizer,
        checkpoint_path=checkpoint_file_path,
        n_epochs=1,
        eval_freq=5,
        eval_iter=5,
        seed=42,
        device="cpu",
    )

    print(
        "Case 1: Temperature 0, top_k None: You are the => ...: ",
        token_ids_to_text(
            generate(
                model,
                token_ids=text_to_token_ids("You are the", tokenizer=tokenizer),
                max_new_tokens=2,
                temperature=0.0,
                top_k=None,
                eos_token_id=50256,
            ),
            tokenizer,
        ),
    )
    print(
        "Case 2: Temperature 2, top_k 5: You are the => ...: ",
        token_ids_to_text(
            generate(
                model,
                token_ids=text_to_token_ids("You are the", tokenizer=tokenizer),
                max_new_tokens=2,
                temperature=2,
                top_k=5,
                eos_token_id=50256,
            ),
            tokenizer,
        ),
    )
