# 《从 Sinusoidal PE 到 RoPE：QK 展开的局限性》

## 1. 为什么需要这篇过渡

如果你已经学完了 Sinusoidal Position Encoding，再直接去看 RoPE，通常会有一个断层：

- 你知道 sinusoidal PE 很优雅
- 你也知道 RoPE 很强
- 但你不一定清楚：**为什么还要从前者走到后者**

真正的过渡点就在 attention 的 `QK` 分数上。

也就是这件事：

```text
score(m, n) = q_m^T k_n
```

问题不是 sinusoidal PE 没有位置信息。  
问题是：

**当 sinusoidal PE 通过“输入相加”的方式进入 attention 后，位置对 `QK` 分数的作用形式不够干净。**

这就是 RoPE 出现的动机。

---

## 2. sinusoidal PE 是怎么进入 attention 的

在原始 Transformer 里，输入是：

```text
x_m = e_m + p_m
```

这里：

- `e_m` 是第 `m` 个 token 的 token embedding
- `p_m` 是第 `m` 个位置的 sinusoidal positional encoding

然后经过线性层得到：

```text
q_m = W_Q x_m = W_Q(e_m + p_m)
k_n = W_K x_n = W_K(e_n + p_n)
```

所以：

```text
q_m = W_Q e_m + W_Q p_m
k_n = W_K e_n + W_K p_n
```

到这里一切都没问题。

---

## 3. 真正关键的一步：把 `q_m^T k_n` 展开

attention 分数是：

```text
score(m, n) = q_m^T k_n
```

把上面的式子代进去：

```text
score(m, n)
= (W_Q e_m + W_Q p_m)^T (W_K e_n + W_K p_n)
```

继续展开：

```text
score(m, n)
= (W_Q e_m)^T (W_K e_n)
 + (W_Q e_m)^T (W_K p_n)
 + (W_Q p_m)^T (W_K e_n)
 + (W_Q p_m)^T (W_K p_n)
```

这 4 项分别代表：

1. 内容和内容的相互作用

```text
(W_Q e_m)^T (W_K e_n)
```

2. 内容和位置的相互作用

```text
(W_Q e_m)^T (W_K p_n)
```

3. 位置和内容的相互作用

```text
(W_Q p_m)^T (W_K e_n)
```

4. 位置和位置的相互作用

```text
(W_Q p_m)^T (W_K p_n)
```

这一步就是过渡的核心。

---

## 4. 第一层局限：位置影响被“混在四项里”

理想情况下，我们希望 attention 分数满足一种很清晰的结构：

```text
score(m, n) = 内容匹配 + 相对位置修正
```

或者至少位置部分能比较明确地写成：

```text
f(n - m)
```

也就是只依赖相对位置。

但在 sinusoidal PE 的输入相加方案里，位置的影响分散在多项里面：

- 有纯位置项
- 有内容和位置的交叉项
- 有位置和内容的交叉项

结果就是：

**位置不是直接进入 attention score 的核心结构，而是绕了一圈、混进了多个项里。**

模型当然可能学会使用这些项，但这个结构本身不够“干净”。

---

## 5. 第二层局限：相对位置没有直接显式出现

很多人会说：

- sinusoidal PE 不是也能表达相对位置吗？

这句话不算错，但要说完整。

确实，sinusoidal PE 有一个很漂亮的性质：

```text
p_{m+k}
```

可以由

```text
p_m
```

通过一个只和 `k` 有关的线性变换得到。

也就是说，**位置编码本身** 对相对位移是友好的。

但注意，这不等于：

**attention 分数本身就自动只依赖 `n - m`。**

因为 attention 真正用的是：

```text
q_m^T k_n
```

而不是直接用：

```text
p_m^T p_n
```

一旦中间经过：

- `x = e + p`
- `q = W_Q x`
- `k = W_K x`

位置结构就和内容结构缠在一起了。

所以你不能直接从“`p_m` 有相对位移性质”跳到“attention score 自然就是相对位置函数”。

这中间还隔着一层很厚的混合。

---

## 6. 第三层局限：交叉项让位置作用依赖内容

看这两项：

```text
(W_Q e_m)^T (W_K p_n)
(W_Q p_m)^T (W_K e_n)
```

它们说明什么？

说明位置对 attention 分数的影响，并不是一个独立、稳定的结构，而会依赖当前 token 的内容表示。

也就是：

- 同一个位置 `n`
- 面对不同内容的 `e_m`

它对 score 的影响可能完全不同。

这件事本身不是“错误”，但它带来一个问题：

**相对位置信息没有以一种结构化、低歧义的方式进入打分公式。**

换句话说，模型要自己在混杂的交叉项里，学出“谁和谁相隔多远”这件事。

这就比“把相对位置直接写进点积结构里”更绕。

---

## 7. 第四层局限：位置编码在输入端注入，离 attention 核心太远

sinusoidal PE 的路径是：

```text
position -> input x -> linear projection -> q/k -> dot product
```

也就是说，位置信息先被加到输入里，再经过一层甚至多层线性变换，最后才影响 attention score。

这个链路的问题在于：

- 位置信息不是直接调控 `q^T k`
- 它只是先进入表示空间，再等待模型自己组织

所以它更像：

**“给模型提示了位置，至于怎么把它变成 attention 的相对关系，要靠模型自己学。”**

而 RoPE 是：

**“直接把位置放进 `q/k` 的几何关系里。”**

这两者在结构上是不一样的。

---

## 8. 一个更直观的对比

### sinusoidal PE 的逻辑

```text
先把位置加到输入上，再由模型自己把这些位置信息揉进 q 和 k
```

对应的问题是：

- 位置和内容混合过早
- `QK` 展开后会出现很多交叉项
- 相对位置没有直接显式写在 score 里

### RoPE 的逻辑

```text
直接对 q 和 k 做按位置的旋转，再去计算点积
```

这样得到：

```text
(R_m q)^T (R_n k) = q^T R(n - m) k
```

于是相对位置直接出现在公式里。

这就是为什么 RoPE 看起来像是对 sinusoidal 思路的一次“结构升级”。

---

## 9. 一个最容易误解的地方

不要把下面两句话混为一谈：

### 句子 A

```text
sinusoidal positional encoding 本身具有相对位移的代数结构
```

这句话是对的。

### 句子 B

```text
把 sinusoidal positional encoding 加到输入后，attention score 就天然只依赖相对位置
```

这句话一般不成立。

因为从：

```text
p_m
```

到：

```text
q_m^T k_n
```

中间经过了内容混合和线性投影，结构已经不再那么纯。

这正是从 Sin 走向 RoPE 时必须看懂的过渡点。

---

## 10. 所以 RoPE 到底解决了什么

RoPE 解决的不是“sinusoidal PE 完全没法用”。

RoPE 真正解决的是：

**把原本在 sinusoidal PE 里比较间接、比较松散的相对位置信号，变成 attention 点积里直接、结构化、可显式写出的相对位置关系。**

也就是把：

```text
位置先加到输入里，再希望模型自己学出来
```

变成：

```text
位置直接写进 q/k 的几何关系中
```

---

## 11. 你现在应该记住的过渡结论

### 1. Sinusoidal PE 不是没有相对位置能力

它的位置向量本身就带有相对位移结构。

### 2. 但这种能力没有被干净地写进 attention score

因为 `q_m^T k_n` 展开后会出现内容项、位置项和交叉项，结构比较混杂。

### 3. RoPE 的价值就在于把相对位置直接嵌入 `QK` 点积

所以它不是推翻 sinusoidal，而是把 sinusoidal 里“旋转/相位差”的数学优点，更直接地放进 attention 核心公式里。

---

## 12. 最后一句总结

从 sinusoidal PE 到 RoPE 的关键过渡，不是“前者错了，后者对了”，而是：

**前者的相对位置能力停留在位置向量层面，后者把这种能力推进到了 attention 打分公式本身。**


