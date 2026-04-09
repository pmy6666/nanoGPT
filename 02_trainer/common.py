from __future__ import annotations

import ast
import math
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


@dataclass
class GPTConfig:
    vocab_size: int
    block_size: int = 128
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 128
    dropout: float = 0.1
    attention_impl: str = "manual"
    gradient_checkpointing: bool = False


@dataclass
class TrainConfig:
    batch_size: int = 16
    max_steps: int = 200
    eval_interval: int = 50
    eval_iters: int = 20
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    warmup_steps: int = 20
    min_lr_ratio: float = 0.1
    grad_clip: float = 1.0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def _parse_scalar(value: str):
    lowered = value.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"null", "none"}:
        return None

    try:
        return ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return value.strip("\"'")


def load_yaml_config(config_path: str | Path | None = None) -> dict:
    path = Path(config_path) if config_path is not None else Path(__file__).resolve().parent / "config.yaml"
    lines = path.read_text(encoding="utf-8").splitlines()

    root: dict = {}
    stack: list[tuple[int, dict]] = [(-1, root)]

    for raw_line in lines:
        if not raw_line.strip() or raw_line.lstrip().startswith("#"):
            continue

        indent = len(raw_line) - len(raw_line.lstrip(" "))
        key, _, raw_value = raw_line.strip().partition(":")
        value = raw_value.strip()

        while stack and indent <= stack[-1][0]:
            stack.pop()

        current = stack[-1][1]
        if not value:
            current[key] = {}
            stack.append((indent, current[key]))
        else:
            current[key] = _parse_scalar(value)

    return root


def load_stage_config(stage_name: str, config_path: str | Path | None = None) -> tuple[GPTConfig, TrainConfig, dict]:
    config = load_yaml_config(config_path)
    stage = config[stage_name]

    gpt_overrides = stage.get("gpt", {})
    train_overrides = stage.get("train", {})
    extras = stage.get("extras", {})

    gpt_config = GPTConfig(**gpt_overrides)
    train_config = TrainConfig(**train_overrides)
    return gpt_config, train_config, extras


class CharTokenizer:
    def __init__(self, text: str):
        chars = sorted(set(text))
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for i, ch in enumerate(chars)}
        self.vocab_size = len(chars)

    def encode(self, s: str) -> list[int]:
        return [self.stoi[c] for c in s]

    def decode(self, ids: list[int]) -> str:
        return "".join(self.itos[i] for i in ids)


def load_text_and_tokenizer(input_path: str | Path | None = None) -> tuple[str, CharTokenizer]:
    base_dir = Path(__file__).resolve().parent.parent
    file_path = Path(input_path) if input_path is not None else base_dir / "input.txt"
    text = file_path.read_text(encoding="utf-8")
    tokenizer = CharTokenizer(text)
    return text, tokenizer


def build_dataset(
    text: str,
    tokenizer: CharTokenizer,
    train_ratio: float = 0.9,
) -> tuple[torch.Tensor, torch.Tensor]:
    data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    split_idx = int(len(data) * train_ratio)
    return data[:split_idx], data[split_idx:]


def get_batch(
    split: str,
    train_data: torch.Tensor,
    val_data: torch.Tensor,
    batch_size: int,
    block_size: int,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    data = train_data if split == "train" else val_data
    starts = torch.randint(0, len(data) - block_size - 1, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in starts])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in starts])
    return x.to(device), y.to(device)


class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head
        self.dropout = config.dropout
        self.attention_impl = config.attention_impl

        self.qkv_proj = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.out_proj = nn.Linear(config.n_embd, config.n_embd)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(config.block_size, config.block_size)).view(
                1, 1, config.block_size, config.block_size
            ),
            persistent=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, channels = x.shape
        qkv = self.qkv_proj(x)
        q, k, v = qkv.split(channels, dim=2)
        q = q.view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1, 2)

        if self.attention_impl == "sdpa":
            attn_out = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=None,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=True,
            )
        else:
            scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
            scores = scores.masked_fill(self.causal_mask[:, :, :seq_len, :seq_len] == 0, float("-inf"))
            probs = F.softmax(scores, dim=-1)
            probs = F.dropout(probs, p=self.dropout, training=self.training)
            attn_out = probs @ v

        attn_out = attn_out.transpose(1, 2).contiguous().view(batch_size, seq_len, channels)
        return self.resid_dropout(self.out_proj(attn_out))


class MLP(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Block(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)
        self.use_checkpoint = config.gradient_checkpointing

    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_checkpoint and self.training:
            return checkpoint(self._forward_impl, x, use_reentrant=False)
        return self._forward_impl(x)


class GPTLanguageModel(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd)
        self.position_embedding = nn.Embedding(config.block_size, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size)
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        batch_size, seq_len = idx.shape
        positions = torch.arange(0, seq_len, device=idx.device)
        x = self.token_embedding(idx) + self.position_embedding(positions)
        x = self.dropout(x)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(batch_size * seq_len, -1), targets.reshape(batch_size * seq_len))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.config.block_size :]
            logits, _ = self(idx_cond)
            next_token_logits = logits[:, -1, :]
            probs = F.softmax(next_token_logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, idx_next], dim=1)
        return idx


def build_optimizer(model: nn.Module, train_config: TrainConfig) -> torch.optim.Optimizer:
    return torch.optim.AdamW(
        model.parameters(),
        lr=train_config.learning_rate,
        weight_decay=train_config.weight_decay,
    )


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    train_config: TrainConfig,
) -> torch.optim.lr_scheduler.LambdaLR:
    def lr_lambda(step: int) -> float:
        if step < train_config.warmup_steps:
            return float(step + 1) / float(max(1, train_config.warmup_steps))

        progress = (step - train_config.warmup_steps) / float(
            max(1, train_config.max_steps - train_config.warmup_steps)
        )
        cosine = 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
        return train_config.min_lr_ratio + (1.0 - train_config.min_lr_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


@torch.no_grad()
def estimate_loss(
    model: nn.Module,
    train_data: torch.Tensor,
    val_data: torch.Tensor,
    train_config: TrainConfig,
    gpt_config: GPTConfig,
) -> dict[str, float]:
    model.eval()
    losses: dict[str, float] = {}
    for split in ("train", "val"):
        split_losses = []
        for _ in range(train_config.eval_iters):
            x, y = get_batch(
                split,
                train_data,
                val_data,
                train_config.batch_size,
                gpt_config.block_size,
                train_config.device,
            )
            _, loss = model(x, y)
            split_losses.append(loss.item())
        losses[split] = sum(split_losses) / len(split_losses)
    model.train()
    return losses
