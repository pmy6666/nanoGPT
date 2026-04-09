import json
import os


class CharTokenizer:
    def __init__(self, text):
        chars = sorted(list(set(text)))
        self.vocab_size = len(chars)
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for i, ch in enumerate(chars)}

    def encode(self, text):
        return [self.stoi[ch] for ch in text]

    def decode(self, ids):
        return ''.join([self.itos[i] for i in ids])


class TiktokenTokenizer:
    def __init__(self, encoding_name='gpt2'):
        try:
            import tiktoken
            self.encoding = tiktoken.get_encoding(encoding_name)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load tiktoken encoding '{encoding_name}'. "
                "Install `tiktoken` and make sure the tokenizer files can be downloaded or are already cached."
            ) from exc
        self.vocab_size = self.encoding.n_vocab

    def encode(self, text):
        return self.encoding.encode(text)

    def decode(self, ids):
        return self.encoding.decode(ids)


def load_text(file_path, encoding='utf-8'):
    with open(file_path, 'r', encoding=encoding) as f:
        return f.read()


def build_tokenizer(text, tokenizer_type='char', encoding_name='gpt2'):
    if tokenizer_type == 'char':
        return CharTokenizer(text)
    if tokenizer_type == 'tiktoken':
        return TiktokenTokenizer(encoding_name)
    raise ValueError(f'Unsupported tokenizer_type: {tokenizer_type}')


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def save_json(data, file_path):
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
