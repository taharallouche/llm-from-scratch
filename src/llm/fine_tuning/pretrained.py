import torch
from safetensors.torch import load_file

from llm.config import GPTConfig
from llm.gpt import GPTModel
from llm.io.utils import checkpoint_path


def load_checkpoint(
    checkpoint_file: str = "gpt2-small-124M.safetensors",
) -> dict[str, torch.Tensor]:
    """Needed to match the parameter names"""
    state_dict = load_file(checkpoint_path(checkpoint_file))

    new_state_dict = {}

    for key, value in state_dict.items():
        key = key.replace(".att.W_query.", ".mh_attention.weight_Q.")
        key = key.replace(".att.W_key.", ".mh_attention.weight_K.")
        key = key.replace(".att.W_value.", ".mh_attention.weight_V.")
        key = key.replace(".att.out_proj.", ".mh_attention.linear_proj.")
        key = key.replace(".att.mask", ".mh_attention.mask")
        key = key.replace(".norm1.", ".layer_norm_1.")
        key = key.replace(".norm2.", ".layer_norm_2.")
        key = key.replace(".ff.layers.", ".ff.seq.")

        new_state_dict[key] = value

    return new_state_dict


def pretrained_gpt_124M() -> GPTModel:
    config = GPTConfig(drop_rate=0.0, qkv_bias=True)
    model = GPTModel(config=config)
    model.load_state_dict(load_checkpoint("gpt2-small-124M.safetensors"))
    return model


def pretrained_gpt_774M() -> GPTModel:
    config = GPTConfig(
        drop_rate=0.0, qkv_bias=True, emb_dim=1280, n_layers=36, n_heads=20
    )
    model = GPTModel(config=config)
    model.load_state_dict(load_checkpoint("gpt2-large-774M.safetensors"))
    return model


if __name__ == "__main__":
    from tokenizers import Tokenizer

    from llm.inference.generate import generate
    from llm.training.pipeline import text_to_token_ids, token_ids_to_text

    tokenizer = Tokenizer.from_pretrained("gpt2")
    text = "I believe I can"
    token_ids = text_to_token_ids(text, tokenizer=tokenizer)

    GPT_774M = pretrained_gpt_774M()

    GPT_774M.eval()

    output = generate(
        model=GPT_774M,
        token_ids=token_ids,
        max_new_tokens=100,
        eos_token_id=50256,
        top_k=10,
        temperature=0.8,
    )
    print(token_ids_to_text(output, tokenizer=tokenizer))
