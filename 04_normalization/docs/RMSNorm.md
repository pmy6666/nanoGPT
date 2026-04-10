# RMSNorm

## 1. 核心目标

RMSNorm 是一种比 LayerNorm 更简洁的归一化方式。

它不减去均值，只使用均方根（root mean square, RMS）来缩放输入。

如果输入为

$$
x \in \mathbb{R}^{B \times D}
$$

对每个样本 \(i\)，定义：

$$
\text{RMS}(x_i) = \sqrt{\frac{1}{D}\sum_{j=1}^{D}x_{i,j}^2 + \varepsilon}
$$

然后归一化：

$$
\hat{x}_{i,j} = \frac{x_{i,j}}{\text{RMS}(x_i)}
$$

最后做逐维缩放：

$$
y_{i,j} = \gamma_j \hat{x}_{i,j}
$$

注意：标准 RMSNorm 通常只有缩放参数 \(\gamma\)，没有平移参数 \(\beta\)。

## 2. 数学直觉

LayerNorm 会先把向量“平移到中心”，再控制尺度。

RMSNorm 只做后一件事：

“我不关心均值是不是 0，我只关心这个向量整体是不是太大或太小。”

它本质上是在控制向量能量：

$$
\frac{1}{D}\sum_j x_{i,j}^2
$$

如果这个量太大，说明整体幅值偏大；如果太小，说明整体幅值偏小。

RMSNorm 把它拉回到稳定范围，但不主动消除均值信息。

## 3. 为什么它比 LayerNorm 更简洁

LayerNorm 使用：

$$
\frac{x-\mu}{\sqrt{\sigma^2+\varepsilon}}
$$

RMSNorm 使用：

$$
\frac{x}{\sqrt{\frac{1}{D}\sum_j x_j^2+\varepsilon}}
$$

差别就在于 RMSNorm 去掉了“减均值”这一步。

这意味着：

- 计算更简单；
- 保留了输入的均值信息；
- 在很多大模型里依然表现很好。

## 4. 配套代码实现

文件：[RMSnorm.py](/Users/pangmy/Desktop/nanoGPT/nanoGPT_component/04_normalization/RMSnorm.py)

```python
rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
x_normalized = x / rms
return self.gamma * x_normalized
```

这里：

- `x.pow(2).mean(dim=-1)` 计算最后一维上的均方；
- 再开方得到 RMS；
- 最后按维度乘 `gamma`。

## 5. 代码实现是否正确

结论：这份实现是正确的最小版 RMSNorm。

它适用于最后一维为 `n_embd` 的输入，例如：

- `[B, D]`
- `[B, T, D]`

我补了维度检查，避免最后一维和参数长度不匹配。

## 6. 数值例子

设输入为：

$$
x = [1, 2, 2]
$$

则均方为：

$$
\frac{1^2 + 2^2 + 2^2}{3} = \frac{9}{3} = 3
$$

因此：

$$
\text{RMS}(x) = \sqrt{3}
$$

归一化后：

$$
\hat{x} = \left[\frac{1}{\sqrt{3}}, \frac{2}{\sqrt{3}}, \frac{2}{\sqrt{3}}\right]
\approx [0.5774, 1.1547, 1.1547]
$$

## 7. 可直接配套的测试样例

```python
import torch
from RMSnorm import RMSNorm

x = torch.tensor([[1.0, 2.0, 2.0]])
rn = RMSNorm(n_embd=3)
y = rn(x)
print(y)
```

预期结果应接近：

```python
tensor([[0.5774, 1.1547, 1.1547]], grad_fn=...)
```

## 8. 和 LayerNorm 的区别

- LayerNorm：减均值，再除标准差。
- RMSNorm：不减均值，只除 RMS。

可以把 RMSNorm 理解成：

“只约束向量长度，不改变向量中心位置。”

这也是它保留更多原始偏置信息的直觉来源。
