from torch import nn

from llm.fine_tuning.pretrained import pretrained_gpt_124M
from llm.gpt import GPTModel

print(pretrained_gpt_124M())


def add_binary_classification_head(
    model: GPTModel, layer: str = "out_head"
) -> GPTModel:
    setattr(model, layer, nn.Linear(in_features=model.emb_dim, out_features=2))
