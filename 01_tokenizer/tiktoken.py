# 调用现有的tiktoken包
import os
import numpy as np
import tiktoken

# 加载训练语料
input_file_path = 'input.txt' 
with open(input_file_path, 'r', encoding='utf-8') as f:
    data = f.read()

print(f"原始字符数: {len(data)}")

# 调用成熟的 GPT-2 Tokenizer
enc = tiktoken.get_encoding("gpt2")
train_ids = enc.encode_ordinary(data) # encode_ordinary 速度更快，忽略特殊 token

print(f"Token 数量: {len(train_ids)}")
print(f"压缩率: {len(data) / len(train_ids):.2f}X")

# 转化为 numpy 数组并保存为 .bin 文件
# GPT-2 的词表大小是 50257，可以用 uint16 (0~65535) 来节省存储空间
train_ids = np.array(train_ids, dtype=np.uint16)
train_ids.tofile('train.bin')
print("数据预处理完成，已保存为 train.bin")