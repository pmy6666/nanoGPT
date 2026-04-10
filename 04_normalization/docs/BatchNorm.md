# Batch Normalization

## 1. 核心目标

BatchNorm 的核心思想是：

对一个 mini-batch 中，同一维特征在样本之间做标准化，使这维特征的数值分布更稳定。

如果输入为

$$
x \in \mathbb{R}^{B \times C}
$$

其中 \(B\) 是 batch size，\(C\) 是特征维度，那么对每个特征维 \(c\)，BatchNorm 计算：

$$
\mu_c = \frac{1}{B}\sum_{i=1}^{B} x_{i,c}
$$

$$
\sigma_c^2 = \frac{1}{B}\sum_{i=1}^{B}(x_{i,c}-\mu_c)^2
$$

然后标准化：

$$
\hat{x}_{i,c} = \frac{x_{i,c}-\mu_c}{\sqrt{\sigma_c^2+\varepsilon}}
$$

最后再做可学习的仿射变换：

$$
y_{i,c} = \gamma_c \hat{x}_{i,c} + \beta_c
$$

## 2. 数学直觉

BatchNorm 在每个特征维上做两件事：

1. 减去均值，让这一维特征在当前 batch 上“居中”。
2. 除以标准差，让这一维特征的波动尺度变得接近 1。

直觉上，它是在问：

“当前样本在这一维上，比这个 batch 的平均水平高多少个标准差？”

所以 BatchNorm 关心的是相对位置，而不是原始绝对大小。

## 3. 为什么要再乘 \(\gamma\)、再加 \(\beta\)

如果只有标准化，模型每层输出都被强行固定成零均值、单位方差，表达能力会受限。

因此加入可学习参数：

$$
\gamma_c,\ \beta_c
$$

让模型自己决定：

- 这一维是否要放大或缩小；
- 这一维是否要整体平移。

也就是说，标准化负责稳定训练，仿射参数负责恢复表达能力。

## 4. 代码对应的数学过程

```python
mean = x.mean(dim=0, keepdim=True)
var = x.var(dim=0, keepdim=True, unbiased=False)
x_normalized = (x - mean) / torch.sqrt(var + self.eps)
return self.gamma * x_normalized + self.beta
```

这里：

- `dim=0` 表示沿 batch 维度统计；
- `keepdim=True` 是为了后续广播；
- `unbiased=False` 对应直接使用
  \(\frac{1}{B}\sum(\cdot)\)，和教学公式一致；
- `gamma`、`beta` 对应上面的 \(\gamma_c\)、\(\beta_c\)。

## 5. 代码实现是否正确

结论：当前实现对二维输入 `x.shape = [batch_size, num_features]` 是正确的。

它实现的是一个“最小可讲解版” BatchNorm，适合教学和理解公式。

我补充了一个输入维度检查，避免形状用错时静默出错。

## 6. 当前实现的边界

这份代码没有实现完整工业版 BatchNorm 的两个常见特性：

1. `running_mean` 和 `running_var`
   推理阶段通常不再使用当前 batch 的统计量，而是使用训练过程累计的运行统计量。
2. 更高维输入支持
   图像中常见的 BatchNorm 往往处理 `NCHW` 张量，而不是只处理 `[B, C]`。

## 7. 简单数值例子

设输入为：

$$
x =
\begin{bmatrix}
1 & 2 \\
3 & 4
\end{bmatrix}
$$

按列做 BatchNorm。

第 1 列：

$$
\mu_1 = \frac{1+3}{2} = 2,\quad
\sigma_1^2 = \frac{(1-2)^2 + (3-2)^2}{2} = 1
$$

标准化后：

$$
\hat{x}_{:,1} = [-1, 1]
$$

第 2 列同理：

$$
\mu_2 = 3,\quad \sigma_2^2 = 1,\quad \hat{x}_{:,2} = [-1, 1]
$$

所以当 (gamma=[1,1], beta=[0,0]) 时：

$$
\hat{x} =
\begin{bmatrix}
-1 & -1 \\
1 & 1
\end{bmatrix}
$$

## 8. 可直接配套的测试样例

```python
import torch
from Batchnorm import BatchNorm

x = torch.tensor([
    [1.0, 2.0],
    [3.0, 4.0],
])

bn = BatchNorm(num_features=2)
y = bn(x)
print(y)
```

预期结果应接近：

```python
tensor([[-1., -1.],
        [ 1.,  1.]], grad_fn=...)
```

会有极小误差，因为分母里加了 `eps`。
