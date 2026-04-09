# BasicTokenizer 的 `encode()` / `decode()` 讲解

## 1. 这份实现的整体定位

[`basic_tokenizer.py`](/Users/pangmy/Desktop/nanoGPT/nanoGPT_code/01_tokenizer/basic_tokenizer.py) 里的 `BasicTokenizer` 是一个教学版 BPE tokenizer。它的目标不是完整复刻生产环境里的 tokenizer，而是把最核心的 3 件事讲清楚：

1. 如何从训练文本里学到 merge 规则。
2. 如何用这些 merge 规则把新文本编码成 token id。
3. 如何把 token id 无损恢复成原始文本。

这版实现的重要前提是：

- 初始基本单位不是字符 `ord(c)`，而是 `utf-8 bytes`。
- `vocab` 保存的是 `token_id -> bytes` 的映射。
- `merges` 保存的是 `(left_id, right_id) -> new_id` 的映射。

这意味着整个 tokenizer 是“字节级、可逆、无损”的。

---

## 2. `encode()` 是怎么实现的

代码位置：

- [`basic_tokenizer.py:58`](/Users/pangmy/Desktop/nanoGPT/nanoGPT_code/01_tokenizer/basic_tokenizer.py#L58)

### 2.1 输入和输出

- 输入：一个 Python 字符串 `text`
- 输出：一个 `list[int]`，即 token id 序列

### 2.2 核心流程

`encode()` 的逻辑可以分成 4 步。

#### 第一步：把字符串转成 utf-8 字节序列

```python
ids = list(text.encode("utf-8"))
```

例如：

```text
"hello" -> [104, 101, 108, 108, 111]
"你" -> [228, 189, 160]
```

这里非常关键。因为 utf-8 是底层表示，所以无论英文、中文、标点还是 emoji，最终都能变成 bytes 序列。

#### 第二步：统计当前序列里所有相邻 pair 的出现情况

```python
stats = self.get_stats(ids)
```

例如序列：

```text
[108, 101, 97, 114, 110, 105, 110, 103]
```

相邻 pair 会是：

```text
(108,101), (101,97), (97,114), ...
```

`get_stats()` 会统计这些 pair 的频率。

#### 第三步：只在“已经学过的 merges”里找当前可合并的 pair

```python
for pair in stats:
    if pair in self.merges:
        idx = self.merges[pair]
        if candidate_idx is None or idx < candidate_idx:
            candidate = pair
            candidate_idx = idx
```

这里不是重新“训练”，而是“应用已有规则”。

这段逻辑的含义是：

- 当前文本里出现了哪些 pair，就检查哪些 pair 在训练阶段被学过。
- 如果有多个 pair 都能合并，就优先使用“更早学到”的 merge。
- 代码里通过 `idx` 更小来近似表示“merge rank 更早”。

这个选择是必要的。否则不同顺序可能得到不同编码结果。

#### 第四步：反复 merge，直到没有可用规则

```python
ids = self.merge(ids, candidate, candidate_idx)
```

只要还能找到可合并的 pair，就继续压缩。最终得到的就是编码结果。

### 2.3 一个直观例子

假设训练阶段已经学到：

```text
(105, 110) -> 258     # "in"
(258, 103) -> 261     # "ing"
```

那么编码 `"ing"` 的过程会是：

1. 初始 bytes: `[105, 110, 103]`
2. 先把 `(105,110)` 合并成 `258`，得到 `[258, 103]`
3. 再把 `(258,103)` 合并成 `261`，得到 `[261]`

最终 `"ing"` 就变成一个 token。

### 2.4 这版 `encode()` 的特点

优点：

- 逻辑直接，能清楚展示 BPE 是怎么工作的。
- 基于 bytes，天然支持任意 utf-8 文本。
- 结果可逆，不会丢信息。

限制：

- 没有做预分词，整段文本直接在字节流上跑 merge。
- 没有 special tokens。
- 没有 offset mapping。
- 没有正则 chunking。
- 编码时每轮都重新统计 pair，效率不高，只适合教学和小规模实验。

---

## 3. `decode()` 是怎么实现的

代码位置：

- [`basic_tokenizer.py:81`](/Users/pangmy/Desktop/nanoGPT/nanoGPT_code/01_tokenizer/basic_tokenizer.py#L81)

### 3.1 输入和输出

- 输入：`list[int]`，token id 序列
- 输出：Python 字符串 `str`

### 3.2 核心流程

`decode()` 只有两步。

#### 第一步：把每个 token id 映射回对应的 bytes

```python
byte_stream = b"".join(self.vocab[idx] for idx in ids)
```

因为训练时维护了：

```python
self.vocab[idx] = self.vocab[pair[0]] + self.vocab[pair[1]]
```

所以每个 merge token 的字节内容都是确定的。

例如：

```text
261 -> b"ing"
265 -> b"fun"
```

那么 token 序列：

```text
[108, 262, 256, 108, 109, 276, 265]
```

会先被还原成一整串 bytes。

#### 第二步：把 bytes 解码成 utf-8 字符串

```python
return byte_stream.decode("utf-8")
```

因为最初编码时就是从 utf-8 bytes 出发，只要 token 序列合法，这一步就能无损恢复原文。

### 3.3 为什么它是可逆的

可逆的原因不是“模型记住了文本”，而是：

1. 基础词表 `0..255` 覆盖了全部单字节。
2. 新 token 只是旧 token 的 bytes 拼接。
3. `vocab` 始终保留了 `token_id -> bytes` 的完整映射。

因此 `decode(encode(text)) == text` 是可以成立的。这也是你做 round-trip 测试时全部通过的根本原因。

---

## 4. 这份实现和主流 LLM tokenizer 的关系

你的实现抓住了主流 tokenizer 的一个核心共性：

- 都需要把文本变成模型能处理的离散 token id
- 都要求 `decode` 基本可逆
- 都试图让高频片段变成更短的 token 序列

但工业实现比教学版多了很多工程层细节。

---

## 5. 当代 LLM 主流 tokenizer 的 `encode()` / `decode()` 是怎么做的

这里先给结论：

### 5.1 主流路线并不只有一种

当前主流大致可以分成三类：

1. 字节级 BPE
2. SentencePiece BPE
3. SentencePiece Unigram

此外，工程实现通常还会叠加：

- 正则或规则化的预切分
- 特殊 token 注入
- 空格处理策略
- offset / span 跟踪
- 跳过 special token 的 decode 选项

---

## 6. 字节级 BPE：OpenAI `tiktoken`、新一代 Llama 的主路线

根据 OpenAI 的 `tiktoken` 仓库，`tiktoken` 是一个给 OpenAI 模型使用的快速 BPE tokenizer；仓库示例也直接展示了 `enc.decode(enc.encode("hello world")) == "hello world"`。Meta 的 `llama-models` 仓库则标明：Llama 2 使用 SentencePiece，而 Llama 3 之后转为 TikToken-based tokenizer。这个变化本身就说明了字节级 BPE 现在非常主流。

### 6.1 这类 tokenizer 的 `encode()` 一般做什么

典型流程不是直接对整段 bytes 做 merge，而是更接近下面这种管线：

1. 先按一套规则或正则把文本切成 chunk。
2. 对每个 chunk 转成 bytes。
3. 在 bytes 上应用 mergeable ranks，也就是已经训练好的 BPE merge 规则。
4. 必要时插入 special tokens。
5. 返回 token ids。

和你的实现相比，差别主要在这里：

- 你是“整段文本直接 bytes -> merge”
- 工业版常常是“先 chunk，再对每个 chunk 做 byte-level BPE”

这样做的好处：

- 空格、标点、英文词干、数字串等边界处理更稳定
- 编码结果与训练分布更一致
- 工程上更快

### 6.2 这类 tokenizer 的 `decode()` 一般做什么

典型流程：

1. 把 token id 映射回 bytes 或字节片段
2. 依次拼接
3. 把拼接结果还原成字符串
4. 根据参数决定是否保留 special tokens

本质上和你的 `decode()` 非常接近。差异主要是：

- 工业版通常区分普通 token 和 special token
- 工业版常支持 `skip_special_tokens=True`
- 工业版会更严格处理无效字节、替代字符和边界情况

### 6.3 为什么字节级 BPE 很适合 LLM

核心原因：

- 覆盖全集。任何输入最终都能表示成 bytes。
- 可逆性强。不会因为词表外字符而彻底失效。
- 多语言更稳。不需要先做强语言假设。

这也是你这份 `BasicTokenizer` 采用 utf-8 bytes 后，设计上已经更接近现代 tokenizer 的原因。

---

## 7. SentencePiece：很多开源模型仍在用

SentencePiece 的官方说明强调两点：

1. 它可以直接从原始句子训练，不一定依赖外部分词。
2. 它支持 BPE 和 Unigram 两种子词算法。

### 7.1 SentencePiece BPE 的 `encode()`

大致流程：

1. 把输入视为 Unicode 字符序列，而不是先手写英语风格分词。
2. 用模型内置规则处理空格和边界。
3. 应用 BPE merges，把字符序列压缩成子词 token。
4. 输出 token ids。

和你的实现相比：

- 你的基本单位是 utf-8 bytes
- SentencePiece 的训练视角更接近 Unicode 字符和子词单元

### 7.2 SentencePiece Unigram 的 `encode()`

这和 BPE 不同，不是“不断合并 pair”，而是：

1. 先准备一个候选子词集合
2. 每个子词有概率或分数
3. 给定输入文本，寻找整体概率更优的切分
4. 输出对应 token ids

这意味着：

- BPE 更像贪心合并
- Unigram 更像在候选词表中做最优切分

### 7.3 SentencePiece 的 `decode()`

大体上仍然是：

1. token id -> piece
2. piece 拼接
3. 恢复空格和边界标记
4. 返回文本

这里的重点不是 bytes 拼接，而是“piece 拼接后如何正确还原空格”。

---

## 8. Hugging Face `tokenizers` 视角下的工业 encode/decode

Hugging Face `tokenizers` 文档把 tokenizer 描述成一个 pipeline。这个表述是准确的。工业 tokenizer 的 `encode()` 往往不是单一步骤，而是管线：

1. Normalizer
2. Pre-tokenizer
3. Model
4. Post-processor

### 8.1 encode 管线

一个现代 tokenizer 的 `encode()` 常见工作包括：

1. 归一化文本
   例如统一空白、大小写、Unicode 规范化。
2. 预分词
   例如按空格、标点、正则模式切块。
3. 子词建模
   例如 BPE、WordPiece、Unigram。
4. 注入 special tokens
   例如 BOS、EOS、系统提示模板相关 token。
5. 生成附加信息
   例如 attention mask、type ids、offset mapping。

你的 `BasicTokenizer.encode()` 实际上只覆盖了其中的第 3 步，而且是教学版的 BPE 子集。

### 8.2 decode 管线

现代 tokenizer 的 `decode()` 通常不仅仅是 `ids -> string`，还会考虑：

- 是否跳过 special tokens
- 是否清理 tokenization spaces
- 是否保留原始空格风格
- 是否输出 offsets 对齐信息

所以工程版 `decode()` 的职责通常比教学版更宽。

---

## 9. 你的实现与主流实现的对应关系

可以把你的实现和工业实现对应起来看：

| 你的实现 | 工业实现中的对应概念 |
| --- | --- |
| `text.encode("utf-8")` | 底层字节表示 |
| `get_stats()` | 统计相邻 pair / 匹配 merge 候选 |
| `merge()` | 应用一条 merge 规则 |
| `self.merges` | merge ranks / merge table |
| `self.vocab[token_id] = bytes` | token id 到字节片段或 piece 的映射 |
| `encode()` 反复 merge | 子词切分过程 |
| `decode()` 拼 bytes | detokenization / string reconstruction |

但工程上还会多出：

- normalizer
- pre-tokenizer
- special tokens
- post-processing
- fast batch encode/decode
- offset tracking

---

## 10. 一个必须说明的技术点

你这份 `encode()` 用的是：

- 每轮重新统计当前 pair
- 然后挑“当前文本里出现且已学习过的 pair 中，token id 最小的那个”

这对教学足够清楚，也能工作；但严格说，它不是所有工业 BPE 实现的唯一写法。

更常见的工业思路是：

- merge 规则有明确 rank
- 每一步选择当前序列里 rank 最优的可合并 pair

你现在的实现里，`idx` 的生成顺序和 merge 学习顺序一致，所以 `idx` 越小，等价于 merge 越早。这就是为什么它在你的代码里成立。

---

## 11. 总结

如果只看核心思想：

- 你的 `encode()` 做的是：`text -> utf-8 bytes -> 按已学 merge 反复压缩 -> token ids`
- 你的 `decode()` 做的是：`token ids -> bytes 拼接 -> utf-8 string`

如果看工业系统：

- 主流 tokenizer 仍然围绕“子词切分 + 可逆解码”
- 但会额外加入 chunking、normalization、special tokens、offset mapping 和更高性能的数据结构

所以这份 `BasicTokenizer` 的价值在于：

- 它已经抓住了现代 tokenizer 最核心的编码/解码本质
- 只是还没有加上生产环境真正依赖的工程层组件

---

## 12. 参考资料

以下资料用于核对“主流 tokenizer 的 encode/decode 做法”和“不同模型家族的路线”：

- OpenAI `tiktoken`: https://github.com/openai/tiktoken
- Hugging Face `tokenizers` API: https://huggingface.co/docs/tokenizers/main/api/tokenizer
- Google `SentencePiece`: https://github.com/google/sentencepiece
- Meta `llama-models`: https://github.com/meta-llama/llama-models

其中“Llama 3 之后采用 TikToken-based tokenizer”来自 Meta `llama-models` 仓库的模型表；这部分是基于该官方仓库信息做的归纳。
