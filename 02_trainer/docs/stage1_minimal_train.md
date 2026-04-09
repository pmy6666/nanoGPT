# 第一阶段：吃透单卡最小 GPT 训练脚本

## 这一阶段要掌握什么

你要能不看资料，自己写出下面这条最小训练链路：

1. `get_batch()` 从一段长文本里切出 `x` 和 `y`
2. `model(x, y)` 做前向传播，得到 `logits` 和 `loss`
3. `optimizer.zero_grad()`
4. `loss.backward()`
5. `clip_grad_norm_()`
6. `optimizer.step()`
7. `scheduler.step()`

## 需要学习的知识点

### 1. 为什么 label 要右移一位

语言模型做的是 next-token prediction。

如果输入是：

```text
x: [我, 爱, 自, 然]
```

标签就应该是：

```text
y: [爱, 自, 然, 语]
```

也就是让模型在看到当前位置以前的 token 后，预测“下一个 token”。

训练时本质上在做：

\[
P(x_2|x_1), P(x_3|x_1,x_2), \dots, P(x_{T+1}|x_1,\dots,x_T)
\]

所以 `get_batch()` 里常见写法是：

```python
x = data[i : i + block_size]
y = data[i + 1 : i + block_size + 1]
```

### 2. 为什么 loss 前要 reshape

模型输出的 `logits` 形状是：

\[
(B, T, V)
\]

其中：

1. `B` 是 batch size
2. `T` 是序列长度
3. `V` 是词表大小

标签 `targets` 的形状是：

\[
(B, T)
\]

而 `F.cross_entropy()` 期望输入是：

1. `input`: `(N, C)`
2. `target`: `(N,)`

所以要把前两个维度合并：

```python
loss = F.cross_entropy(
    logits.reshape(B * T, V),
    targets.reshape(B * T),
)
```

直觉上，你可以把它理解成：

“原来有 `B` 个样本、每个样本里有 `T` 个位置，现在把所有位置都摊平成 `B*T` 个分类任务，一次性算交叉熵。”

### 3. 为什么要做 gradient clipping

反向传播后，某些 step 的梯度可能突然很大，尤其是：

1. 学习率偏大
2. 序列较长
3. 模型更深
4. 混合精度训练时数值更脆弱

如果直接 `optimizer.step()`，参数更新会过猛，导致：

1. loss 突然爆炸
2. 训练不稳定
3. 出现 `nan`

常见写法：

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

它不是“让训练更快”，而是“限制一次更新别太离谱”。

### 4. 为什么 scheduler 要在 optimizer 后面更新

推荐顺序：

```python
loss.backward()
clip_grad_norm_(...)
optimizer.step()
scheduler.step()
```

原因是这个 step 的梯度，应该用“当前学习率”先更新参数；更新完成后，再把学习率推进到“下一个 step”。

否则就会出现：

“本来这个 step 应该用 lr(step)，结果你提前切到了 lr(step+1)。”

很多 PyTorch scheduler 也是按这个语义设计的。

## 你应该能顺口解释的训练链路

### dataloader / batch

从长文本中随机截取长度为 `block_size` 的连续片段，`x` 是原片段，`y` 是右移一位后的目标片段。

### forward

`idx -> token embedding + position embedding -> Transformer blocks -> lm_head -> logits`

### loss

每个位置都做一次“在词表上的多分类”，最后把 `B*T` 个位置的 loss 聚合成一个标量。

### backward

`loss.backward()` 会把这个标量 loss 对模型参数的梯度算出来，并累积到每个参数的 `.grad` 上。

### optimizer step

优化器读取 `.grad`，按 AdamW 规则更新参数。

## 建议你自己动手做的练习

1. 把 `block_size` 从 `128` 改成 `32`，观察 loss 和生成结果变化。
2. 把 `grad_clip` 改成 `0.1`、`1.0`、`5.0`，感受梯度裁剪强度的区别。
3. 先注释掉 `scheduler.step()`，再恢复，观察训练前期收敛速度。
4. 在 `get_batch()` 里打印一组 `x[0]` 和 `y[0]`，确保你真的理解“右移一位”。
