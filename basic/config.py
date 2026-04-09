from dataclasses import dataclass

import torch


@dataclass
class GPTConfig:
    batch_size: int = 64
    block_size: int = 256
    max_iters: int = 5000
    eval_interval: int = 500
    eval_iters: int = 200
    log_interval: int = 500
    learning_rate: float = 3e-4
    n_embd: int = 384
    n_head: int = 6
    n_layer: int = 6
    dropout: float = 0.2
    train_split: float = 0.9
    seed: int = 1337
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer_type: str = "tiktoken"    # char ot tiktoken
    tokenizer_encoding: str = "gpt2"
    data_file: str = "input.txt"
    out_dir: str = "out"
    checkpoint_name: str = "basic_gpt.pt"
    sample_prompt: str = "Hello world"
    sample_tokens: int = 100
    use_flash: bool = True      # use flash attention
