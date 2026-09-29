from pathlib import Path

import tiktoken

TOKENIZER = tiktoken.get_encoding("gpt2")


encodings = TOKENIZER.encode(Path("the-verdict.txt").read_text())
