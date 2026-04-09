from dataclasses import dataclass

import torch
from torch.utils.data import Dataset

from common import GPTLanguageModel, load_stage_config, load_text_and_tokenizer


try:
    from transformers import Trainer, TrainingArguments
except ImportError as exc:
    raise ImportError("Install transformers before running this example.") from exc


@dataclass
class LanguageModelingSample:
    input_ids: torch.Tensor
    labels: torch.Tensor


class CharDataset(Dataset):
    def __init__(self, text: str, tokenizer, block_size: int):
        self.data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
        self.block_size = block_size

    def __len__(self) -> int:
        return len(self.data) - self.block_size - 1

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        x = self.data[idx : idx + self.block_size]
        y = self.data[idx + 1 : idx + self.block_size + 1]
        return {"input_ids": x, "labels": y}


class TrainerWrappedGPT(GPTLanguageModel):
    def forward(self, input_ids: torch.Tensor, labels: torch.Tensor | None = None):
        logits, loss = super().forward(input_ids, labels)
        return {"loss": loss, "logits": logits}


def main() -> None:
    text, tokenizer = load_text_and_tokenizer()
    gpt_config, train_config, extras = load_stage_config("stage3_hf_trainer_style")
    block_size = gpt_config.block_size

    train_cut = int(len(text) * 0.9)
    train_dataset = CharDataset(text[:train_cut], tokenizer, block_size)
    eval_dataset = CharDataset(text[train_cut:], tokenizer, block_size)

    gpt_config.vocab_size = tokenizer.vocab_size
    model = TrainerWrappedGPT(gpt_config)

    args = TrainingArguments(
        output_dir="hf_trainer_runs",
        per_device_train_batch_size=train_config.batch_size,
        per_device_eval_batch_size=train_config.batch_size,
        learning_rate=train_config.learning_rate,
        weight_decay=train_config.weight_decay,
        warmup_steps=train_config.warmup_steps,
        max_steps=train_config.max_steps,
        eval_steps=extras["eval_steps"],
        logging_steps=extras["logging_steps"],
        save_steps=extras["save_steps"],
        evaluation_strategy="steps",
        lr_scheduler_type="cosine",
        report_to="none",
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        fp16=torch.cuda.is_available() and not torch.cuda.is_bf16_supported(),
        gradient_accumulation_steps=1,
    )

    trainer = Trainer(model=model, args=args, train_dataset=train_dataset, eval_dataset=eval_dataset)
    trainer.train()


if __name__ == "__main__":
    main()
