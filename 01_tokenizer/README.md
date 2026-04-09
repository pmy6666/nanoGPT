# 01_tokenizer

这个目录用于从零理解 tokenizer 的核心思路，重点放在 BPE 和实际工程中的现成 tokenizer 用法对比。

## 目录说明

- `basic_tokenizer.py`
  一个玩具版 BPE tokenizer，实现了训练、编码、解码和 merge 规则查看，适合用来理解 tokenizer 是如何从字节级别逐步学出更长子词的。

- `BPE.py`
  一个更短的小脚本，用最少代码演示 BPE 的核心循环：
  1. 统计相邻 token 对频率
  2. 找到最高频 pair
  3. 合并为新的 token
  4. 重复执行直到达到 merge 次数

- `tiktoken.py`
  演示如何直接使用成熟库 `tiktoken` 对文本进行编码，并保存为后续训练可用的 `train.bin`。

- `docs/`
  放 tokenizer 相关的学习笔记，包括 `BPE`、`WordPiece`、`Unigram` 和 `encode/decode` 的说明。

## 学习顺序建议

建议按下面顺序阅读和运行：

1. 先看 `BPE.py`
   目标是理解 BPE 最基础的 merge 过程。

2. 再看 `basic_tokenizer.py`
   目标是理解一个最小可用 tokenizer 需要哪些能力：
   - 训练 merge 规则
   - `encode`
   - `decode`
   - round-trip 测试

3. 最后看 `tiktoken.py`
   目标是理解实际项目里为什么常常直接复用成熟 tokenizer，而不是自己从零实现。

## 运行方式

在项目根目录下运行：

```bash
python nanoGPT_component/01_tokenizer/BPE.py
python nanoGPT_component/01_tokenizer/basic_tokenizer.py
python nanoGPT_component/01_tokenizer/tiktoken.py
```

## 代码要点

### `basic_tokenizer.py`

核心流程如下：

- 初始状态下，把文本转成 UTF-8 字节序列
- 词表初始只包含 `0~255` 的单字节 token
- 每一轮统计所有相邻 token 对出现次数
- 选择最高频 pair，分配一个新 token id
- 更新词表和 merge 规则
- 编码时按已学习到的 merge 顺序不断压缩
- 解码时把 token 对应的字节流拼接回原文

这个实现适合学习，但还不是工业级 tokenizer。它没有处理更复杂的预分词、特殊 token、并行性能和大规模语料训练问题。

### `tiktoken.py`

这个脚本展示了一个更贴近训练工程的流程：

- 读取原始语料 `input.txt`
- 使用 GPT-2 tokenizer 进行编码
- 将 token id 保存成 `uint16` 的二进制文件

这一步通常是训练语言模型前的数据预处理步骤。

## 输出结果可以关注什么

运行这些脚本时，可以重点观察：

- 文本长度是否因为 merge 而变短
- 高频子串是否会被学成独立 token
- `encode -> decode` 是否能无损还原原文
- 现成 tokenizer 的 token 数量和压缩效果

## 依赖说明

- `BPE.py` 和 `basic_tokenizer.py` 只依赖 Python 标准库
- `tiktoken.py` 需要额外安装：

```bash
pip install tiktoken numpy
```

## 备注

如果你是在学习 nanoGPT 或 GPT 类模型，这个目录的作用可以理解为：

- `BPE.py` 解决“tokenizer 的核心思想是什么”
- `basic_tokenizer.py` 解决“最小可用 tokenizer 怎么写”
- `tiktoken.py` 解决“真实训练中通常怎么做”
