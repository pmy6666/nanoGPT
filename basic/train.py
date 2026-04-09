import os
import time
from dataclasses import asdict

import torch

from config import GPTConfig
from model import GPTLanguageModel
from utils import build_tokenizer, ensure_dir, load_text, save_json


config = GPTConfig()
torch.manual_seed(config.seed)


def build_dataset(cfg):
    text = load_text(cfg.data_file)
    tokenizer = build_tokenizer(
        text,
        tokenizer_type=cfg.tokenizer_type,
        encoding_name=cfg.tokenizer_encoding,
    )
    dataset = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    split_idx = int(cfg.train_split * len(dataset))
    train_data = dataset[:split_idx]
    val_data = dataset[split_idx:]
    return text, tokenizer, train_data, val_data


def get_batch(split, train_data, val_data, cfg):
    data = train_data if split == "train" else val_data
    if len(data) <= cfg.block_size:
        raise ValueError(
            f"{split} dataset is too short for block_size={cfg.block_size}. "
            "Reduce block_size or use a larger dataset."
        )
    ix = torch.randint(len(data) - cfg.block_size, (cfg.batch_size,))
    x = torch.stack([data[i : i + cfg.block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + cfg.block_size + 1] for i in ix])
    return x.to(cfg.device), y.to(cfg.device)


@torch.no_grad()
def estimate_loss(model, train_data, val_data, cfg):
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(cfg.eval_iters)
        for k in range(cfg.eval_iters):
            xb, yb = get_batch(split, train_data, val_data, cfg)
            _, loss = model(xb, yb)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


def save_checkpoint(model, optimizer, tokenizer, cfg, out_dir):
    checkpoint_path = os.path.join(out_dir, cfg.checkpoint_name)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": asdict(cfg),
            "vocab_size": tokenizer.vocab_size,
        },
        checkpoint_path,
    )
    return checkpoint_path


def sync_device(cfg):
    if cfg.device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


def run_training():
    text, tokenizer, train_data, val_data = build_dataset(config)
    model = GPTLanguageModel(config, tokenizer.vocab_size).to(config.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    tokens_per_step = config.batch_size * config.block_size

    print(f"device: {config.device}")
    print(f"tokenizer: {config.tokenizer_type}")
    print(f"vocab_size: {tokenizer.vocab_size}")
    print(f"dataset_tokens: {len(train_data) + len(val_data)}")
    print(f"model_parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    print(f"tokens_per_step: {tokens_per_step}")

    for step in range(config.max_iters):
        if step % config.eval_interval == 0:
            losses = estimate_loss(model, train_data, val_data, config)
            print(
                f"step {step}: train loss {losses['train']:.4f}, "
                f"val loss {losses['val']:.4f}"
            )

        sync_device(config)
        step_start = time.perf_counter()
        xb, yb = get_batch("train", train_data, val_data, config)
        _, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        sync_device(config)
        step_time = time.perf_counter() - step_start
        tokens_per_sec = tokens_per_step / step_time if step_time > 0 else float("inf")
        print(
            f"step {step}: loss {loss.item():.4f}, "
            f"step_time {step_time * 1000:.2f} ms, "
            f"throughput {tokens_per_sec:.2f} tokens/s"
        )
        

    ensure_dir(config.out_dir)
    save_json(asdict(config), os.path.join(config.out_dir, "train_config.json"))
    checkpoint_path = save_checkpoint(
        model,
        optimizer,
        tokenizer,
        config,
        config.out_dir,
    )

    prompt_ids = torch.tensor(
        tokenizer.encode(config.sample_prompt),
        dtype=torch.long,
        device=config.device,
    ).unsqueeze(0)
    generated = model.generate(prompt_ids, max_new_tokens=config.sample_tokens)[0].tolist()
    print("sample:")
    print(tokenizer.decode(generated))
    print(f"checkpoint saved to: {checkpoint_path}")


if __name__ == "__main__":
    run_training()
