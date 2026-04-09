import re
from collections import Counter, defaultdict

def get_stats(ids):
    """统计相邻 token 对的频率"""
    counts = Counter()
    for i in range(len(ids) - 1):
        counts[(ids[i], ids[i+1])] += 1
    return counts

def merge(ids, pair, idx):
    """将序列中所有的 pair 替换为新的 idx"""
    new_ids = []
    i = 0
    while i < len(ids):
        if i < len(ids) - 1 and (ids[i], ids[i+1]) == pair:
            new_ids.append(idx)
            i += 2
        else:
            new_ids.append(ids[i])
            i += 1
    return new_ids

# 模拟
text = "learning llm is fun, learning pytorch is also fun"
tokens = list(text.encode("utf-8")) # 初始为 ASCII 码
vocab = {i: bytes([i]) for i in range(256)} # 初始词表  dict = (int -> char/string)
num_merges = 10
merges = {} # 记录合并规则： dict = ((p1, p2) -> new_idx)

for i in range(num_merges):
    stats = get_stats(tokens)
    if not stats: break
    pair = max(stats, key=stats.get)
    idx = 256 + i
    tokens = merge(tokens, pair, idx)
    merges[pair] = idx
    vocab[idx] = vocab[pair[0]] + vocab[pair[1]]
    print(f"合并 {pair} 为 {idx}, 频率: {stats[pair]}")

print(f"最终序列长度从 {len(text)} 压缩到了 {len(tokens)}")