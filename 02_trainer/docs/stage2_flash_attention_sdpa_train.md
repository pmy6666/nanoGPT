# 第二阶段：Flash Attention / SDPA

## 需要学习的知识点

### 先说结论

PyTorch 里的 `scaled_dot_product_attention` 是统一接口。

当硬件、dtype、shape 合适时，它底层可能调用：

1. Flash Attention kernel
2. memory-efficient attention kernel
3. math kernel

所以学习时要区分：

1. `SDPA` 是接口
2. `Flash Attention` 是一种更高效的 attention 实现算法/内核

## 数学公式原理

标准 self-attention：

```text
Q = X · W_Q
K = X · W_K
V = X · W_V
```

```text
S = (QK^T) / sqrt(d_k)
```

加 causal mask 后：

```text
S_tilde[i, j] = S[i, j],     if j <= i
S_tilde[i, j] = -inf,        if j > i
```

然后做 softmax：

```text
P = softmax(S_tilde)
```

最后：

```text
O = P · V
```

标准实现里，问题出在中间矩阵：

1. `S` 形状是 `(B, H, T, T)`
2. `P` 形状也是 `(B, H, T, T)`

当序列长度 `T` 很大时，这两个张量特别吃显存。

## 直观直觉

### 传统实现像什么

像先把整张大表格全部写在显存里：

1. 先把每个 query 对所有 key 的分数都算出来
2. 整张分数表存下来
3. softmax 后再存一张概率表
4. 再去和 `V` 相乘

也就是：

“先完整展开，再计算。”

### Flash Attention 像什么

像分块流式处理：

1. 不把完整 `T x T` attention matrix 落到显存
2. 只处理一个 tile
3. 一边算局部 softmax，一边维护数值稳定所需的统计量
4. 直接把结果往输出方向累积

也就是：

“边读、边算、边归一化、边写输出。”

因此它能显著减少 HBM 显存读写。

## 为什么它更省显存

主要省的是：

1. attention score matrix `QK^T`
2. softmax 后的概率矩阵
3. 大量中间临时张量

本质上，它避免了把 `(T, T)` 级别的大矩阵完整 materialize 到显存。

## 为什么它通常更快

attention 往往不是纯算力瓶颈，而是显存带宽瓶颈。

Flash Attention 通过减少 HBM 读写，让更多工作在更快的片上存储里完成，所以常常同时得到：

1. 更低显存
2. 更高吞吐

## 对训练稳定性有没有副作用

理论目标没有变，还是在算同一个 attention。

但实际工程里要注意：

1. 不同 kernel 的数值路径略有差异
2. 不同 dtype 下误差特征会不同
3. 某些特殊 mask / shape / dropout 组合可能退回别的 kernel

一般来说它不是“让训练更不稳定”的主因，但你要知道不同实现的浮点误差不可能完全一致。

## 应该写在训练脚本哪里

训练循环几乎不用改，核心改的是模型里的 attention 实现：

```python
attn_out = F.scaled_dot_product_attention(
    q,
    k,
    v,
    attn_mask=None,
    dropout_p=self.dropout if self.training else 0.0,
    is_causal=True,
)
```

也就是说：

1. `optimizer`
2. `scheduler`
3. `backward`

这些位置都不需要特殊改。

改的是 `forward` 里 attention 的那一段。

## 你要能回答的关键问题

### 它到底省的是哪块显存

不是主要省参数显存，而是省 attention 中间结果显存。

### 它会不会一定更快

不一定。

如果：

1. 硬件不支持合适 kernel
2. dtype 不合适
3. shape 太小
4. 最终退回 math kernel

那收益就可能不明显。

### 它和 gradient checkpointing 有什么区别

1. Flash Attention 优化的是 attention 算法本身
2. checkpointing 优化的是 activation 保存策略

二者可以一起用。

## 建议练习

1. 先写一版手搓 attention，再切到 `scaled_dot_product_attention`。
2. 比较长序列时两种实现的显存峰值。
3. 观察不同 dtype 下，SDPA 是否能带来更明显收益。
