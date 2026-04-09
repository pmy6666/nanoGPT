# 《RoPE 底层原理》

## 1. 先说结论：RoPE 到底在做什么

RoPE 的全称是 **Rotary Position Embedding**。

它和普通的“把位置向量加到输入上”不一样。

RoPE 的核心想法是：

**不是把位置直接加到 token 表示上，而是把位置编码进 `query` 和 `key` 的旋转里。**

也就是说，RoPE 不再主要回答：

- “这个 token 的位置向量是多少”

而是更进一步回答：

- “当这个 token 去和别的 token 做 attention 时，它的位置应该怎样影响匹配关系”

这就是它和 sinusoidal PE 的最大差别。

---

## 2. 为什么还要在 sinusoidal PE 之后再搞一个 RoPE

Sinusoidal Position Encoding 已经很好了，它有两个优点：

- 固定公式，不需要学习
- 对相对位置有很好的代数结构

但它的位置注入方式是：

```text
x_input = token_embedding + positional_encoding
```

也就是：

- 先把位置信息加进输入
- 再让后面的线性层和 attention 自己去学怎么用

RoPE 更直接。它的思路是：

**既然 attention 最终依赖的是 `q` 和 `k` 的点积，那我就直接让“位置”进入这个点积结构本身。**

所以 RoPE 的设计目标非常清楚：

**让 attention score 天然带上相对位置信息。**

---

## 3. 先回顾 attention 里真正决定“关注谁”的量

attention 的核心分数是：

```text
score(m, n) = q_m^T k_n
```

这里：

- `m` 是 query 所在位置
- `n` 是 key 所在位置

如果我们希望 attention 理解顺序，一个很自然的愿望就是：

**这个分数最好能和相对位置 `m - n` 有明确关系。**

RoPE 就是在做这件事。

---

## 4. RoPE 的最小数学单元：二维旋转

先只看二维向量：

```text
[x1, x2]
```

RoPE 会把它按位置 `m` 旋转一个角度 `theta_m`：

```text
R(theta_m) [x1, x2]
```

其中二维旋转矩阵是：

```text
R(theta) = [
  [ cos(theta), -sin(theta)],
  [ sin(theta),  cos(theta)]
]
```

也就是说，原本的二维向量没有变成别的东西，而是：

**在平面里转了一下。**

如果位置不同，旋转角度就不同。

于是：

- 位置 `m` 的 token，会旋转 `theta_m`
- 位置 `n` 的 token，会旋转 `theta_n`

---

## 5. 多维情况：每两维构成一个旋转子空间

真实模型里向量不是 2 维，而是很多维。

RoPE 的做法是：

- 第 0、1 维是一组
- 第 2、3 维是一组
- 第 4、5 维是一组
- ...

每一组都做一次二维旋转，但**使用的频率不同**。

所以你可以把一个 `d` 维向量理解成：

```text
[pair_0, pair_1, pair_2, ...]
```

每个 `pair_i` 都会按照自己的角频率旋转。

这和 sinusoidal PE 很像，因为它们都在用“多频率的二维结构”；  
但 RoPE 把这个结构直接用在 `q/k` 上，而不是单独构造一个位置向量再去相加。

---

## 6. RoPE 的核心公式

对于第 `i` 个二维 pair，定义角频率：

```text
omega_i = 1 / 10000^(2i / d)
```

那么位置 `m` 对应的旋转角度就是：

```text
theta_{m,i} = m * omega_i
```

于是，RoPE 对 `q_m` 和 `k_n` 的处理可以写成：

```text
q'_m = R_m q_m
k'_n = R_n k_n
```

这里：

- `R_m` 表示“按位置 `m` 的各组角度组成的分块旋转矩阵”
- `R_n` 表示“按位置 `n` 的各组角度组成的分块旋转矩阵”

最后 attention 分数变成：

```text
score(m, n) = (R_m q_m)^T (R_n k_n)
```

这就是 RoPE 的入口公式。

---

## 7. 最关键的数学性质：attention 分数只和相对位置有关

这是你必须真正吃透的地方。

因为旋转矩阵满足：

```text
R_m^T R_n = R(n - m)
```

所以：

```text
(R_m q_m)^T (R_n k_n)
= q_m^T R_m^T R_n k_n
= q_m^T R(n - m) k_n
```

这一步非常重要。

它说明经过 RoPE 之后，attention 分数里的位置部分，不再分别依赖 `m` 和 `n`，而是自然收缩成：

```text
n - m
```

也就是：

**attention score 天然带上了相对位移信息。**

这正是 RoPE 最强的地方。

它不是“模型可能学到相对位置”，而是“位置结构直接进入了点积公式”。

---

## 8. 数学直觉：为什么旋转会产生相对位置

这个直觉可以这样理解：

- query 在位置 `m`，先转一下
- key 在位置 `n`，也转一下

当你再去算它们的内积时，本质上比较的不是两个绝对角度，而是两个角度之间的差。

而“角度之差”对应的就是：

```text
theta_n - theta_m
```

由于 `theta = pos * omega`，所以它又等价于：

```text
(n - m) * omega
```

也就是相对位置。

你可以把它想成两个指针：

- 每个位置都会把自己的向量指针转一点
- attention 看两个指针有多对齐

那它自然更关心“它们相差多少角度”，而不是“它们分别转了多少圈”

---

## 9. 为什么说 RoPE 是对 sinusoidal PE 的自然升级

你可以把两者的关系理解成下面这句话：

**sinusoidal PE 是把位置写成一组旋转坐标；RoPE 是直接让 `q/k` 在这些旋转坐标里参与 attention。**

两者共同点：

- 都有多频率结构
- 都按二维 pair 组织
- 都和 `sin/cos` 有关
- 都对相对位置有良好的数学性质

区别在于注入位置的地方不同：

- sinusoidal PE：在输入端相加
- RoPE：在 attention 里的 `q/k` 上旋转

所以 RoPE 更“贴近 attention 本体”。

---

## 10. 复数视角下看 RoPE，会更简单

把二维向量：

```text
[x1, x2]
```

看成一个复数：

```text
z = x1 + i x2
```

那么旋转一个角度 `theta`，就等价于乘以：

```text
e^{i theta}
```

于是 RoPE 就变成：

```text
z' = z * e^{i theta}
```

位置不同，只是乘上的相位不同。

而两个位置之间的相互作用，自然就会出现：

```text
e^{i(theta_n - theta_m)}
```

也就是相位差。

这个复数视角本质上和前面的二维旋转矩阵是同一件事，只是写法更紧凑。

---

## 11. 代码实现时到底做了什么

工程里一般不会真的构造一个大旋转矩阵 `R`，因为那样太笨重。

通常做法是：

1. 先把向量按偶数维 / 奇数维两两分组
2. 为每组维度算出当前位置对应的 `cos` 和 `sin`
3. 用下面这个局部公式做旋转

对一个 pair `[x_even, x_odd]`：

```text
x_rot_even = x_even * cos(theta) - x_odd * sin(theta)
x_rot_odd  = x_even * sin(theta) + x_odd * cos(theta)
```

这就是二维旋转矩阵乘法展开后的结果。

---

## 12. 为什么 RoPE 通常只作用在 q 和 k，不作用在 v

因为 attention score 是由 `q` 和 `k` 的点积决定的：

```text
score = q^T k
```

RoPE 的目标是把相对位置信息注入这个“匹配分数”里。

所以最关键的是改 `q` 和 `k`。

而 `v` 主要负责“被聚合的内容”，它不参与分数计算本身，因此通常不需要做同样的旋转。

---

## 13. 你真正要记住的三个结论

### 1. RoPE 不是把位置向量加到输入上

它是把位置编码成 `q/k` 的旋转。

### 2. 它最重要的数学收益，是让 attention 分数天然依赖相对位置

因为：

```text
(R_m q)^T (R_n k) = q^T R(n - m) k
```

### 3. 它本质上是“多频率二维旋转”

每两维构成一个平面，在不同频率下按位置旋转。

---

## 14. 配套代码应该重点看什么

配套代码会演示 4 件事：

1. 如何对向量应用 RoPE 旋转
2. 为什么旋转后的 attention 分数只和相对位置有关
3. 同时平移两个位置时，attention 分数为什么不变
4. RoPE 和 sinusoidal PE 在二维 pair 结构上是怎么对应起来的

---

## 15. 最后一句总结

RoPE 的本质不是“再发明一种位置编码”，而是：

**把位置信息直接写进 attention 的点积结构里，让相对位置关系以旋转相位差的形式自然出现。**
