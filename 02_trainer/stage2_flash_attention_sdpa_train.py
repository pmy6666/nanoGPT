import torch

from common import (
    GPTLanguageModel,
    build_dataset,
    build_optimizer,
    build_scheduler,
    estimate_loss,
    get_batch,
    load_text_and_tokenizer,
    load_stage_config,
)


def main() -> None:
    torch.manual_seed(42)

    text, tokenizer = load_text_and_tokenizer()
    train_data, val_data = build_dataset(text, tokenizer)

    gpt_config, train_config, _ = load_stage_config("stage2_flash_attention_sdpa_train")
    gpt_config.vocab_size = tokenizer.vocab_size

    model = GPTLanguageModel(gpt_config).to(train_config.device)
    optimizer = build_optimizer(model, train_config)
    scheduler = build_scheduler(optimizer, train_config)

    for step in range(train_config.max_steps):
        if step % train_config.eval_interval == 0:
            losses = estimate_loss(model, train_data, val_data, train_config, gpt_config)
            print(f"step {step:04d} | train loss {losses['train']:.4f} | val loss {losses['val']:.4f}")

        x, y = get_batch(
            "train",
            train_data,
            val_data,
            train_config.batch_size,
            gpt_config.block_size,
            train_config.device,
        )

        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), train_config.grad_clip)
        optimizer.step()
        scheduler.step()

        if step % 20 == 0:
            print(
                f"step {step:04d} | loss {loss.item():.4f} | "
                f"grad_norm {grad_norm:.4f} | attention_impl {gpt_config.attention_impl}"
            )


if __name__ == "__main__":
    main()
