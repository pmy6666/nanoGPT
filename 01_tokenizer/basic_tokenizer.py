"""
    使用BPE算法实现的玩具版 tokenizer
"""
from collections import Counter


class BasicTokenizer:
    def __init__(self):
        """
        作用:
            初始化 tokenizer 的核心状态,包括合并规则、词表和合并历史。
        Input:
            无
        Output:
            无,创建一个可训练和可编码/解码的 BasicTokenizer 实例。
        """
        self.merges = {}
        self.vocab = {i: bytes([i]) for i in range(256)}
        self.merge_history = []

    @staticmethod
    def get_stats(ids):
        """
        作用:
            统计 token 序列中所有相邻 token 对出现的频率。
        Input:
            ids: list[int],一个 token id 序列。
        Output:
            Counter: key是相邻 token 对 `(id1, id2)`, value是该 token 对出现的次数。
        """
        counts = Counter()
        for i in range(len(ids) - 1):
            counts[(ids[i], ids[i + 1])] += 1
        return counts

    @staticmethod
    def merge(ids, pair, idx):
        """
        作用:
            将序列中所有连续出现的目标 token 对替换成一个新的 token id。
        Input:
            ids: list[int],原始 token id 序列。
            pair: tuple[int, int],需要被合并的相邻 token 对。
            idx: int, 新生成的 token id。
        Output:
            list[int],完成一次 merge 后的新 token id 序列。
        """
        new_ids = []
        i = 0
        while i < len(ids):
            if i < len(ids) - 1 and (ids[i], ids[i + 1]) == pair:
                new_ids.append(idx)
                i += 2
            else:
                new_ids.append(ids[i])
                i += 1
        return new_ids

    def train(self, text, num_merges):
        """
        作用:
            在输入文本上训练 BPE 规则,学习若干条高频 merge, 并更新词表。
        Input:
            text: str,用来训练 tokenizer 的原始文本。
            num_merges: int,最多执行多少次 merge。
        Output:
            list[int],训练文本在完成所有 merge 后得到的压缩 token 序列。
        """
        ids = list(text.encode("utf-8"))
        self.merges = {}
        self.vocab = {i: bytes([i]) for i in range(256)}
        self.merge_history = []

        for i in range(num_merges):
            stats = self.get_stats(ids)
            if not stats:
                break

            pair = max(stats, key=stats.get)
            if stats[pair] < 2:
                break

            idx = 256 + i
            ids = self.merge(ids, pair, idx)
            self.merges[pair] = idx
            self.vocab[idx] = self.vocab[pair[0]] + self.vocab[pair[1]]
            self.merge_history.append(
                {
                    "rank": i + 1,
                    "pair": pair,
                    "idx": idx,
                    "count": stats[pair],
                    "token_bytes": self.vocab[idx],
                }
            )

        return ids

    def encode(self, text):
        """
        作用:
            使用已经学到的 merge 规则,把输入文本编码成 token id 序列。
        Input:
            text: str,需要编码的原始文本。
        Output:
            list[int],编码后的 token id 序列。
        """
        ids = list(text.encode("utf-8"))

        while len(ids) >= 2:
            stats = self.get_stats(ids)
            candidate = None
            candidate_idx = None

            for pair in stats:
                if pair in self.merges:
                    idx = self.merges[pair]
                    if candidate_idx is None or idx < candidate_idx:
                        candidate = pair
                        candidate_idx = idx

            if candidate is None:
                break

            ids = self.merge(ids, candidate, candidate_idx)

        return ids

    def decode(self, ids):
        """
        作用:
            将 token id 序列还原成 utf-8 字节流,再解码回原始字符串。
        Input:
            ids: list[int],需要解码的 token id 序列。
        Output:
            str,解码后的文本。
        """
        byte_stream = b"".join(self.vocab[idx] for idx in ids)
        return byte_stream.decode("utf-8")

    def print_top_merges(self, k=10):
        """
        作用:
            按学习顺序打印前 k 条 merge,便于检查学到的子词是否合理。
        Input:
            k: int,最多打印多少条 merge 记录,默认是 10。
        Output:
            无,直接把 merge 信息打印到终端。
        """
        print(f"\nTop {min(k, len(self.merge_history))} merges:")
        for item in self.merge_history[:k]:
            token_text = item["token_bytes"].decode("utf-8", errors="replace")
            print(
                f"#{item['rank']:02d} idx={item['idx']} "
                f"pair={item['pair']} count={item['count']} token={token_text!r}"
            )


def run_round_trip_tests(tokenizer, samples):
    """
    作用:
        对多条样例执行 encode -> decode 的 round-trip 测试,检查是否能还原原文。
    Input:
        tokenizer: BasicTokenizer, 已经训练完成的 tokenizer 实例。
        samples: list[str],需要测试的文本样例列表。
    Output:
        无,直接打印每条样例的测试结果以及整体是否通过。
    """
    print("\nRound-trip tests:")
    all_passed = True

    for i, sample in enumerate(samples, start=1):
        encoded = tokenizer.encode(sample)
        decoded = tokenizer.decode(encoded)
        passed = decoded == sample
        all_passed = all_passed and passed
        print(
            f"[{i}] {'PASS' if passed else 'FAIL'} | "
            f"text={sample!r} | tokens={encoded}"
        )

    print(f"\nAll round-trip tests passed: {all_passed}")


if __name__ == "__main__":
    train_text = (
        "learning llm is fun, learning pytorch is also fun.\n"
        "Byte Pair Encoding works on utf-8 bytes.\n"
        "Tokenizer learning should support English, numbers like 12345,\n"
        "and Chinese like 你好,世界。\n"
        "Repeated phrases help BPE learn useful merges: learning learning fun fun.\n"
    )

    tokenizer = BasicTokenizer()
    compressed_ids = tokenizer.train(train_text, num_merges=30)

    print(f"Original byte length: {len(train_text.encode('utf-8'))}")
    print(f"Compressed token length after training: {len(compressed_ids)}")

    samples = [
        "hello",
        "learning llm is fun",
        "learning pytorch is also fun",
        "Byte Pair Encoding",
        "utf-8 bytes",
        "numbers: 12345",
        "你好",
        "你好,世界。",
    ]
    run_round_trip_tests(tokenizer, samples)
    tokenizer.print_top_merges(k=10)
