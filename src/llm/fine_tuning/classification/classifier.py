import tiktoken
import torch
from torch import nn
from torch.utils.data import DataLoader

from llm.fine_tuning.pretrained import pretrained_gpt_124M
from llm.gpt import GPTModel
from llm.inference.generate import generate
from llm.io.paper_reviews import create_dataloaders
from llm.training.pipeline import text_to_token_ids, token_ids_to_text


def setup_binary_classification_fine_tuning(
    model: GPTModel,
    out_layer: str = "out_head",
    trf_blocks_layer: str | None = None,
    final_norm_layer: str | None = None,
) -> GPTModel:
    for param in model.parameters():
        param.requires_grad = False

    setattr(
        model, out_layer, nn.Linear(in_features=model.config.emb_dim, out_features=2)
    )

    if trf_blocks_layer:
        trf_blocks: nn.Sequential = getattr(model, trf_blocks_layer)
        for param in trf_blocks[-1].parameters():
            param.requires_grad = True

    if final_norm_layer:
        final_norm: nn.Module = getattr(model, final_norm_layer)
        for param in final_norm.parameters():
            param.requires_grad = True

    return model


def compute_loss_batch(
    input_batch: torch.Tensor, target_batch: torch.Tensor, model: GPTModel, device
) -> float:
    logits = model(input_batch)  # n_batch x seq_length x 2

    last_logits = logits[:, -1, :]  # n_batch x 2

    return torch.nn.functional.cross_entropy(last_logits, target_batch.squeeze()).item()


def compute_loss(
    dataloader: DataLoader,
    model: GPTModel,
    max_batches: int | None = None,
    device="cpu",
) -> float:

    total_losses = 0

    for batches_seen, (input_batch, target_batch) in enumerate(dataloader, start=1):
        batch_loss = compute_loss_batch(input_batch, target_batch, model, device)

        total_losses += batch_loss

        if max_batches and batches_seen == max_batches:
            break

    return total_losses / batches_seen


if __name__ == "__main__":
    tokenizer = tiktoken.get_encoding("gpt2")
    input_ = text_to_token_ids("This is a test", tokenizer=tokenizer)

    model = setup_binary_classification_fine_tuning(pretrained_gpt_124M())

    train, val, test = create_dataloaders()

    with torch.no_grad():
        loss = compute_loss(train, model)

    print("Loss: ", loss)
