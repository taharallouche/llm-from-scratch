import torch

inputs = torch.tensor(
    [
        [0.43, 0.15, 0.89],  # Your
        [0.55, 0.87, 0.66],  # journey
        [0.57, 0.85, 0.64],  # starts
        [0.22, 0.58, 0.33],  # with
        [0.77, 0.25, 0.10],  # one
        [0.05, 0.80, 0.55],  # step
    ]
)


attention_scores = inputs @ inputs.transpose(0, 1)
attention_weights = attention_scores / attention_scores.sum(dim=1).unsqueeze(-1)


softmaxed_attention_weights = torch.softmax(attention_scores, dim=1)


# cij = sum(wik xkj)
context_vec = softmaxed_attention_weights @ inputs

print(context_vec)
