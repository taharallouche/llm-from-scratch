import torch
from tiktoken import Encoding

from dataloader import create_dataloader, read_the_verdict
from tokenizer import TOKENIZER


def vocab_size(tokenizer: Encoding) -> int:
    return tokenizer.max_token_value + 1


output_dim = 256
token_embedding_layer = torch.nn.Embedding(
    num_embeddings=vocab_size(TOKENIZER), embedding_dim=output_dim
)

# The length of the input sequence, needs to equal to max_length in dataloader
context_length = 4

pos_embedding_layer = torch.nn.Embedding(
    num_embeddings=context_length, embedding_dim=output_dim
)
pos_embedding = pos_embedding_layer(torch.arange(context_length))


dataloader = create_dataloader(
    txt=read_the_verdict(), batch_size=8, max_length=4, stride=4
)

data_iter = iter(dataloader)

batch_inputs, batch_targets = next(data_iter)
batch_inputs: torch.Tensor

print(f"Batch input shape: {batch_inputs.shape}")


token_embeddings = token_embedding_layer(batch_inputs)

print(f"Embedding layer shape: {token_embeddings.shape}")


input_embeddings = token_embeddings + pos_embedding

print(f"Final embeddings shape: {token_embeddings.shape}")
