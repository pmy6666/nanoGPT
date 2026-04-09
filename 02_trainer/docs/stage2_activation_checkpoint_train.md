# 第二阶段：Gradient Checkpointing

## 需要学习的知识点

### 它省的是哪部分显存

主要省的是：

1. 中间激活值
2. block 内部前向结果缓存

正常反向传播时，Autograd 需要保存很多中间结果，方便 backward 直接用。

gradient checkpointing 的做法是：

“前向时少存一点，反向时需要了再重新算一遍。”

所以它本质上是在用算力换显存（时间换空间）。

### 会不会变慢

会，通常会变慢。

因为 backward 时会额外重算一遍被 checkpoint 的那段前向。

### 对训练稳定性的副作用

它本身一般不改变数学结果，主要副作用不是“不稳定”，而是：

1. 训练时间变长
2. 某些带随机性的层要留意复现性
3. 和某些自定义算子组合时需要检查兼容性

### 应该写在训练脚本哪里

它最核心的改动其实不在训练循环，而在模型定义里。

例如 Transformer block：

```python
def forward(self, x):
    if self.use_checkpoint and self.training:
        return checkpoint(self._forward_impl, x, use_reentrant=False)
    return self._forward_impl(x)
```

训练循环基本可以不变。

## 关键理解

### 什么叫“用算力换显存”

原来：

1. 前向存很多中间结果
2. 反向直接读这些缓存

现在：

1. 前向少存
2. 反向时重新执行一遍那段前向，拿到需要的中间值

所以显存下降了，但时间上升了。

### 最适合 checkpoint 哪些模块

通常是最深、最耗激活的重复结构：

1. Transformer block
2. 大 MLP
3. attention block

## 建议练习

1. 先只给后半部分 block 开 checkpoint，再给全部 block 开。
2. 对比同一 batch size 下显存峰值差异。
3. 对比 step time，感受它的“慢”主要来自哪里。
