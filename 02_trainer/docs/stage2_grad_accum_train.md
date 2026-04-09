# 第二阶段：梯度累积

## 需要学习的知识点

### 梯度累积省的是哪部分显存

它本质上不是压缩单次 forward 的激活，而是：

“把大 batch 拆成多个小 micro-batch 依次跑。”

因此节省的是：

1. 单次前向/反向需要承载的 activation 峰值
2. 单次 batch 输入张量占用

它不直接减少：

1. 模型参数显存
2. 优化器状态显存

### 会不会变慢

通常会变慢一点。

原因是原本一个大 batch 可以并行做完，现在变成多次前向和多次反向，kernel launch 次数更多。

但如果你的卡放不下大 batch，那梯度累积是最实用的折中方案。

### 对训练稳定性的副作用

如果你记得把 loss 除以 `grad_accum_steps`，理论上它和大 batch 训练更接近。

如果忘了除，会导致梯度放大 `grad_accum_steps` 倍，相当于偷偷增大学习率，训练很容易炸。

### 应该写在训练脚本哪里

它改的是“一个 optimizer step 内部”的结构：

```python
optimizer.zero_grad(set_to_none=True)
for micro_step in range(grad_accum_steps):
    _, loss = model(x, y)
    loss = loss / grad_accum_steps
    loss.backward()

clip_grad_norm_(...)
optimizer.step()
scheduler.step()
```

## 核心理解

### effective batch size

\[
\text{effective batch size} = \text{micro batch size} \times \text{grad accum steps}
\]

例如：

1. micro-batch = 8
2. grad_accum_steps = 4

那么等效 batch size = 32。

### 为什么要最后再 step

因为你想先把 4 个 micro-batch 的梯度累起来，再执行一次参数更新。

## 建议练习

1. 把 `grad_accum_steps` 从 `4` 改到 `8`，观察吞吐变化。
2. 删掉 `loss = loss / grad_accum_steps`，体会 loss 为什么会不稳定。
3. 记录一次 optimizer step 之前的梯度范数，看看累积后会怎么变。
