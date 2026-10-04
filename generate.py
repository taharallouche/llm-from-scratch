import torch

from gpt import GPTModel


def generate(
    model: GPTModel,
    token_ids: torch.Tensor,  # n_batch x seq_length
    max_new_tokens: int,
    temperature: float,
    top_k: int | None = None,
) -> torch.Tensor:
    """
    Assumes
    """

    model.eval()
    for _ in range(max_new_tokens):
        with torch.no_grad():
            logits: torch.Tensor = model(
                token_ids[:, -model.context_size :]
            )  # n_batch x context_length x vocab_size

            logits = logits[:, -1, :]  # n_batch x vocab_size

        if top_k is not None:
            top_k_logits, _ = torch.topk(logits, k=top_k, dim=-1, sorted=True)

            min_val_per_seq = top_k_logits[:, -1].unsqueeze(-1)

            logits = logits.where(logits < min_val_per_seq, 0)

        if temperature > 0:
            logits = logits / temperature
            probas = torch.softmax(
                logits, dim=-1
            )  # n_batch x vocab_size (seq, token_proba)
            next_tokens_ids = torch.multinomial(probas, num_samples=1)  # n_batch x 1

        else:
            next_tokens_ids = torch.argmax(
                logits, dim=-1, keepdim=True
            )  # n_batch x 1 (seq, next token id)

        token_ids = torch.concat([token_ids, next_tokens_ids], dim=-1)

    return token_ids
