from torch import nn

from llm.gpt import GPTModel


def setup_binary_classification_model(
    model: GPTModel,
    out_layer: str = "out_head",
) -> GPTModel:
    setattr(
        model,
        out_layer,
        nn.Linear(
            in_features=model.config.emb_dim,
            out_features=2,
        ),
    )
    return model


def setup_binary_classification_fine_tuning(
    model: GPTModel,
    out_layer: str = "out_head",
    trf_blocks_layer: str | None = None,
    final_norm_layer: str | None = None,
) -> GPTModel:
    model = setup_binary_classification_model(model=model, out_layer=out_layer)

    for param in model.parameters():
        param.requires_grad = False

    if trf_blocks_layer:
        trf_blocks: nn.Sequential = getattr(model, trf_blocks_layer)
        for param in trf_blocks[-1].parameters():
            param.requires_grad = True

    if final_norm_layer:
        final_norm: nn.Module = getattr(model, final_norm_layer)
        for param in final_norm.parameters():
            param.requires_grad = True

    getattr(model, out_layer).requires_grad_(True)

    return model
