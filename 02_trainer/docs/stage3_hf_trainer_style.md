# 第三阶段：Hugging Face Trainer 风格

## 需要学习的知识点

### Trainer 帮你接管了什么

它会帮你统一管理：

1. dataloader
2. train / eval loop
3. optimizer / scheduler
4. logging
5. checkpoint save/load
6. AMP、梯度累积等常见功能开关

### 你自己还必须知道什么

即使你用了 Trainer，也必须能说清：

1. 样本是怎么构造成 `input_ids` 和 `labels` 的
2. 模型 `forward` 为什么要返回 `loss`
3. `gradient_accumulation_steps` 真正影响的是哪里
4. 学习率调度器在 step 之间怎么推进

## 为什么这一阶段要放到后面学

因为 Trainer 容易让人只会“调参数”，不会“理解训练”。

如果你前两阶段没吃透，看到：

```python
trainer = Trainer(...)
trainer.train()
```

很容易以为训练就是一个黑盒。

## 你要重点观察的映射关系

### 你手写训练循环里的什么，映射到了 Trainer 的哪里

1. `get_batch()` -> `Dataset` / `DataCollator`
2. `model(x, y)` -> `forward(input_ids, labels)`
3. `optimizer.step()` -> Trainer 内部封装
4. `scheduler.step()` -> Trainer 内部封装
5. AMP / grad accumulation -> `TrainingArguments`

## 建议练习

1. 先把 `stage1_minimal_train.py` 和这份脚本逐行对照。
2. 尝试自己写一个 `data_collator`，不要完全依赖默认行为。
3. 明确 Trainer 帮你“藏”起来的那几步，自己口头复述出来。
