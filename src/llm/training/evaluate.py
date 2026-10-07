import torch
from torch.utils.data import DataLoader

from llm.gpt import GPTModel


def compute_batch_loss(
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


def compute_loss(
    dataloader: DataLoader, model: GPTModel, device, num_batches: int | None = None
) -> float:
    total_loss = 0
    batch_count = 0
    for iter, (input_batch, target_batch) in enumerate(dataloader):
        if num_batches is not None and iter == num_batches:
            break

        batch_loss = compute_batch_loss(
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


def evaluate_model(
    model: GPTModel,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    eval_iter: int | None = None,
    device="cpu",
) -> tuple[float, float]:
    model.eval()
    with torch.no_grad():
        return compute_loss(
            train_dataloader, model, device, num_batches=eval_iter
        ), compute_loss(val_dataloader, model, device, num_batches=eval_iter)
