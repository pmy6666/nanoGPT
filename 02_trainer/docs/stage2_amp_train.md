# 第二阶段：AMP 混合精度

## 需要学习的知识点

### AMP 省的是哪部分显存

主要省的是：

1. 前向激活值
2. 部分中间张量
3. attention / matmul 的临时结果

因为很多算子会从 `float32` 降到 `float16` 或 `bfloat16`，所以激活通常能接近减半。

注意：

1. 参数主副本不一定都减半，取决于实现
2. AdamW 的优化器状态通常仍然很占显存
3. AMP 最主要节省的是 activation，不是 optimizer state

### 会不会变慢

通常在现代 GPU 上会更快，尤其是 Tensor Core 友好的矩阵乘法。

但有两个例外：

1. 很小的模型上，加速不明显
2. 如果频繁发生数值回退、cast 开销偏多，也可能收益有限

### 对训练稳定性的副作用

`float16` 可能更容易数值下溢，所以通常要配合 `GradScaler`。

`bfloat16` 动态范围更大，训练通常比 `float16` 更稳，但显存优势和吞吐表现要看硬件。

### 应该写在训练脚本哪里

核心插入点有三个：

1. forward 外面套 `autocast`
2. backward 前用 `scaler.scale(loss)`
3. `clip_grad_norm_()` 前先 `scaler.unscale_(optimizer)`

典型顺序：

```python
optimizer.zero_grad(set_to_none=True)
with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
    _, loss = model(x, y)

scaler.scale(loss).backward()
scaler.unscale_(optimizer)
clip_grad_norm_(...)
scaler.step(optimizer)
scaler.update()
scheduler.step()
```

## 你要能回答的面试式问题

### 为什么 clip grad 前要先 unscale

因为 `GradScaler` 缩放过 loss，梯度也被一起放大了。

如果你不先 `unscale_()`，裁剪的是“放大后的假梯度”，阈值就失真了。

### AMP 最适合优化哪段

最吃矩阵乘法的部分：

1. attention 的 QK^T、AV
2. MLP 的线性层

## 建议练习

1. 把 `amp_dtype` 从 `float16` 改成 `bfloat16`，比较稳定性。
2. 故意删掉 `scaler.unscale_(optimizer)`，理解为什么 clip 会不对。
3. 统计训练中 `torch.cuda.max_memory_allocated()`，看 AMP 前后差异。
