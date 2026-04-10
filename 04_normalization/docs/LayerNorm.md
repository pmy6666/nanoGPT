# Layer Normalization

## 1. 核心目标

LayerNorm 的核心思想是：

对单个样本内部的特征维进行标准化，而不是在 batch 内跨样本统计。

如果输入为

$$
x \in \mathbb{R}^{B \times D}
$$

对每个样本 \(i\)，LayerNorm 在该样本自己的最后一维上计算：

$$
\mu_i = \frac{1}{D}\sum_{j=1}^{D} x_{i,j}
$$

$$
\sigma_i^2 = \frac{1}{D}\sum_{j=1}^{D}(x_{i,j}-\mu_i)^2
$$

然后标准化：

$$
\hat{x}_{i,j} = \frac{x_{i,j}-\mu_i}{\sqrt{\sigma_i^2+\varepsilon}}
$$

再做逐维仿射变换：

$$
y_{i,j} = \gamma_j \hat{x}_{i,j} + \beta_j
$$

## 2. 数学直觉

BatchNorm 问的是：

“在这个 batch 里，这个样本的某一维比别人高多少？”

LayerNorm 问的是：

“在这个样本内部，这一维比它自己的其他维高多少？”

所以 LayerNorm 不依赖 batch 统计量，这也是它特别适合 Transformer 的原因之一。

即使 batch size 很小，甚至为 1，它依然可以正常工作，因为它只看当前样本自身。

## 3. 为什么适合序列模型

在 NLP/Transformer 中，输入通常是：

$$
x \in \mathbb{R}^{B \times T \times D}
$$

其中 \(D\) 是 embedding 维度。

LayerNorm 一般沿最后一维 \(D\) 做归一化，因此每个 token 都独立完成标准化：

$$
\mu_{b,t} = \frac{1}{D}\sum_{j=1}^{D}x_{b,t,j}
$$

这意味着不同 batch、不同时间步之间互不影响。

## 4. 你的代码对应的数学过程

文件：[Layernorm.py](/Users/pangmy/Desktop/nanoGPT/nanoGPT_component/04_normalization/Layernorm.py)

```python
mean = x.mean(dim=-1, keepdim=True)
var = x.var(dim=-1, keepdim=True, unbiased=False)
x_normalized = (x - mean) / torch.sqrt(var + self.eps)
return self.gamma * x_normalized + self.beta
```

这里：

- `dim=-1` 表示沿最后一维归一化；
- `gamma`、`beta` 的长度等于最后一维大小；
- 广播规则会把它们应用到每个样本上。

## 5. 代码实现是否正确

结论：当前实现是正确的。

它适用于最后一维大小等于 `n_embd` 的输入，例如：

- `[B, D]`
- `[B, T, D]`

我补充了一个最后一维的形状检查，避免 embedding 维度不匹配。

## 6. 数值例子

设单个样本为：

$$
x = [1, 2, 3]
$$

则：

$$
\mu = \frac{1+2+3}{3} = 2
$$

$$
\sigma^2 = \frac{(1-2)^2 + (2-2)^2 + (3-2)^2}{3} = \frac{2}{3}
$$

所以标准化结果为：

$$
\hat{x} =
\left[
\frac{-1}{\sqrt{2/3}},
0,
\frac{1}{\sqrt{2/3}}
\right]
\approx
[-1.2247, 0, 1.2247]
$$

## 7. 可直接配套的测试样例

```python
import torch
from Layernorm import LayerNorm

x = torch.tensor([[1.0, 2.0, 3.0]])
ln = LayerNorm(n_embd=3)
y = ln(x)
print(y)
```

预期结果应接近：

```python
tensor([[-1.2247,  0.0000,  1.2247]], grad_fn=...)
```

## 8. 和 BatchNorm 的关键区别

- BatchNorm：对同一特征维，在 batch 中跨样本统计。
- LayerNorm：对同一样本，在特征维内部统计。

因此：

- BatchNorm 依赖 batch；
- LayerNorm 不依赖 batch；
- Transformer 中通常更偏向 LayerNorm。
