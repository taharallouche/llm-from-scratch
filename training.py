from functools import partial
from pathlib import Path

import tiktoken
import torch
from torch.utils.data import DataLoader

from dataloader import create_dataloader
from generate import generate
from gpt import GPTConfig, GPTModel
from tokenizer import TOKENIZER

SMALL_CONTEXT_CONFIG = GPTConfig(context_length=256)


def text_to_token_ids(text: str, tokenizer: tiktoken.Encoding) -> torch.Tensor:
    token_ids = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
    return torch.tensor(token_ids, dtype=torch.long).unsqueeze(0)


def token_ids_to_text(
    token_ids: torch.Tensor, tokenizer: tiktoken.Encoding
) -> list[str]:
    return tokenizer.decode(token_ids.squeeze(0).tolist())


### Loss


def calc_loss_batch(
    input_batch: torch.Tensor,
    target_batch: torch.Tensor,
    model: GPTModel,
    device,
) -> torch.Tensor:
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)
    logits: torch.Tensor = model(input_batch)  # n_batch x context_length x vocab_size
    flattened_logits = logits.flatten(0, 1)  # (n_batch * context_length) x vocab size
    loss = torch.nn.functional.cross_entropy(
        input=flattened_logits, target=target_batch.flatten()
    )

    return loss


def calc_loss_loader(
    dataloader: DataLoader, model: GPTModel, device, num_batches: int | None = None
) -> float:
    total_loss = 0
    batch_count = 0
    for iter, (input_batch, target_batch) in enumerate(dataloader):
        if num_batches is not None and iter == num_batches:
            break

        batch_loss = calc_loss_batch(
            input_batch=input_batch,
            target_batch=target_batch,
            model=model,
            device=device,
        )

        total_loss += batch_loss.item()
        batch_count += 1

    if batch_count == 0:
        return float("nan")

    return total_loss / batch_count


### Training Loop


def evaluate_model(
    model: GPTModel,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    eval_iter: int,
    device,
) -> float:
    model.eval()
    with torch.no_grad():
        return calc_loss_loader(
            train_dataloader, model, device, num_batches=eval_iter
        ), calc_loss_loader(val_dataloader, model, device, num_batches=eval_iter)


def train_model(
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    model: GPTModel,
    optimizer: torch.optim.Optimizer,
    n_epochs: int,
    eval_freq: int,
    eval_iter: int,
    device,
):
    global_step = -1

    train_losses, val_losses = [], []
    for epoch in range(n_epochs):
        for input_ids, target_ids in train_dataloader:
            model.train()

            batch_loss = calc_loss_batch(input_ids, target_ids, model, device=device)
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

                print(
                    f"Ep {epoch + 1} (Step {global_step:06d}): "
                    f"Train loss {train_loss:.3f}, "
                    f"Val loss {val_loss:.3f}"
                )

    return train_losses, val_losses


if __name__ == "__main__":
    #### Data Preparation
    text_data = Path("the-verdict.txt").read_text()
    num_chars = len(text_data)
    token_ids = tiktoken.get_encoding("gpt2").encode(text_data)
    num_tokens = len(token_ids)

    train_ratio = 0.9

    split_idx = int(train_ratio * num_chars)

    train_data, val_data = text_data[:split_idx], text_data[split_idx:]

    dataloader = partial(
        create_dataloader,
        batch_size=2,
        max_length=SMALL_CONTEXT_CONFIG.context_length,
        stride=SMALL_CONTEXT_CONFIG.context_length,
        num_workers=0,
    )

    train_dataloader = dataloader(
        txt=train_data,
        drop_last=True,
        shuffle=True,
    )
    val_dataloader = dataloader(txt=val_data, drop_last=False, shuffle=False)

    ### Training
    torch.manual_seed(42)
    model = GPTModel(SMALL_CONTEXT_CONFIG)
    optimizer = torch.optim.AdamW(params=model.parameters())
    train_losses, val_losses = train_model(
        train_dataloader,
        val_dataloader,
        model=model,
        optimizer=optimizer,
        n_epochs=10,
        eval_freq=5,
        eval_iter=5,
        device="cpu",
    )
    print(
        "Temperature 0, top_k None",
        generate(
            model,
            token_ids=text_to_token_ids("You are the", tokenizer=TOKENIZER),
            max_new_tokens=2,
            temperature=0.0,
            top_k=None,
        ),
    )
    print(
        "Temperature 2, top_k 5",
        generate(
            model,
            token_ids=text_to_token_ids("You are the", tokenizer=TOKENIZER),
            max_new_tokens=2,
            temperature=2,
            top_k=5,
        ),
    )
