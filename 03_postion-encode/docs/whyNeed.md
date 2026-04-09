# 《为什么 Transformer 需要位置编码》

## 这件事到底在解决什么问题

Transformer 里的 self-attention 很强，但它有一个先天缺口：

它会看 token 和 token 之间“彼此相关不相关”，却**不会天然知道谁在前、谁在后**。

所以，位置编码要解决的问题不是“让模型看到更多内容”，而是：

**把序列中的顺序信息显式告诉 Transformer。**

如果不做这一步，Transformer 看到的更像是一个“token 集合”，而不是一个“有先后顺序的句子”。

---

## 先回顾 self-attention 公式

```text
Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V
```

这里：

- `QK^T` 表示 query 和 key 的相似度打分
- `sqrt(d_k)` 是缩放项
- `softmax(...)` 会把打分变成注意力权重
- 最后再用这些权重对 `V` 做加权求和

它的核心流程可以粗暴理解成：

1. 每个 token 先映射成 `Q`、`K`、`V`
2. 用 `QK^T` 计算“谁应该关注谁”
3. 用 `softmax` 得到注意力权重
4. 对所有 `V` 做加权求和，得到当前位置的新表示

这里最关键的一点是：

**这个公式本身只关心 token 内容之间的相似性，不关心 token 在原句子里的绝对位置或相对顺序。**

---

## 为什么 attention 本身不感知顺序

假设一句话有 3 个 token：

```text
[x1, x2, x3]
```

它们经过线性变换后得到：

```text
[q1, q2, q3], [k1, k2, k3], [v1, v2, v3]
```

attention 只会计算：

```text
q1 和所有 k 的相似度
q2 和所有 k 的相似度
q3 和所有 k 的相似度
```

问题在于，`q1`、`q2`、`q3` 如果只来自 token embedding，那么它们只携带“这个词是什么”，不携带“这个词在第几个位置”。

于是模型知道：

- 有 `dog`
- 有 `bites`
- 有 `man`

但它**不知道**：

- 是 `dog` 在前、`man` 在后
- 还是 `man` 在前、`dog` 在后

换句话说，**不加位置信息时，self-attention 对 token 的排列是置换不敏感的**。  
你把序列重排，attention 看到的内容集合几乎还是同一批 token。

---

## 如果没有位置编码，Transformer 会丢失什么

它会丢失最关键的 **顺序信息**，具体包括：

### 1. 绝对位置

模型不知道某个 token 是第 1 个、第 5 个，还是第 20 个。

例如：

- 句首 token
- 句尾 token
- 当前词前面已经出现了多少词

这些信息都会缺失。

### 2. 相对位置

模型不知道两个 token 谁在前谁在后，也不知道它们相隔多远。

例如：

- 主语在动词前面还是后面
- 修饰词离被修饰词有多远
- 当前词应该更关注前一个词还是前十个词

这些都属于相对位置信息。

### 3. 由顺序决定的语义结构

自然语言不是“词袋”，而是“有顺序的结构”。

如果没有顺序，很多语义关系都会塌掉，比如：

- 谁是动作发出者
- 谁是动作承受者
- 否定词修饰的是哪个部分
- 时间、条件、转折修饰的是哪一段

---

## 例子：为什么 `"dog bites man"` 和 `"man bites dog"` 语义完全不同

这两个句子的 token 集合几乎一样：

```text
dog, bites, man
man, bites, dog
```

差别只在顺序。

但语义完全不同：

- `dog bites man`：狗咬人
- `man bites dog`：人咬狗

为什么会这样？

因为语言里的语义不只由“有哪些词”决定，还由“这些词按什么顺序组织”决定。

在这个例子里：

- 排在动词 `bites` 前面的词，更像施事者
- 排在动词 `bites` 后面的词，更像受事者

如果 Transformer 没有位置编码，它可能只能看见：

- 有一个 `dog`
- 有一个 `man`
- 有一个 `bites`

却分不清到底是谁咬谁。

这就是为什么：

**token 集合相似，不代表语义相似；顺序本身就是意义的一部分。**

---

## token embedding 和 position encoding 分别负责什么

这是最容易混淆、但必须分清的一点。

### token embedding 负责什么

`token embedding` 负责把“离散的 token id”变成“连续的向量表示”。

它主要编码的是：

- 这个词是什么
- 这个词和其他词在语义上像不像
- 这个词在训练中学到的统计特征

例如：

- `dog` 和 `cat` 的 embedding 可能更接近
- `run` 和 `walk` 的 embedding 可能更接近

所以你可以把 token embedding 理解成：

**词的内容表示、语义表示。**

### position encoding 负责什么

`position encoding` 负责告诉模型：

- 这个 token 在第几个位置
- 两个 token 谁在前谁在后
- 它们相距多远

所以你可以把 position encoding 理解成：

**词的位置信息、顺序信息。**

### 两者合起来才完整

Transformer 输入端常见做法是：

```text
输入表示 = token embedding + position encoding
```

含义是：

- `token embedding` 说“这是什么词”
- `position encoding` 说“它在什么位置”

只有两者合起来，模型才知道：

**这个词是什么，以及它出现在句子的哪里。**

---

## 为什么 RNN / CNN 天然带顺序，而 Transformer 不带

### 为什么 RNN 天然带顺序

RNN 按时间步一个一个处理 token：

```text
x1 -> h1
x2 -> h2
x3 -> h3
```

其中：

```text
h_t = f(x_t, h_{t-1})
```

这意味着当前状态 `h_t` 一定依赖前一个状态 `h_{t-1}`，也就天然依赖“前面已经看过什么”。

所以 RNN 的计算路径本身就是有方向、有先后的，顺序已经写进计算图里了。

### 为什么 CNN 天然带顺序

CNN 虽然并不是按时间步递推，但它在序列上做卷积时，会用局部窗口：

```text
[x1, x2, x3]
[x2, x3, x4]
```

卷积核扫过不同位置时，位置不同就意味着接收到的局部上下文不同。

也就是说，CNN 的输出依赖“这个特征出现在序列的哪个局部区域”，因此也天然保留了一部分位置信息。

### 为什么 Transformer 不天然带顺序

Transformer 的 self-attention 是“所有 token 两两直接交互”的结构。

它不会像 RNN 那样按时间推进，也不会像 CNN 那样用固定局部窗口从左到右扫描。

如果输入里不额外加入位置，attention 只会看到一组 token 表示之间的相互匹配关系，而不会知道这些 token 的排列顺序。

所以：

- RNN：顺序写在递推过程中
- CNN：顺序写在局部卷积结构里
- Transformer：顺序必须额外注入

---

### 1. token embedding 负责什么

token embedding 负责把 token id 映射成向量，表示词的内容和语义特征，回答的是“这个 token 是什么”。

### 2. position encoding 负责什么

position encoding 负责把位置信息加入输入表示，告诉模型 token 在序列中的位置以及 token 之间的先后关系，回答的是“这个 token 在哪里”。

### 3. 为什么 RNN/CNN 天然带顺序，而 Transformer 不带

RNN 通过时间递推天然编码顺序，CNN 通过局部卷积窗口天然保留位置信息；但 Transformer 的 self-attention 只建模 token 之间的相关性，不自带顺序感，因此必须额外加入位置编码。

---

## 一句话收尾

**没有位置编码，Transformer 就很难区分“同一组词的不同排列”；而在自然语言里，顺序本身就是意义。**
