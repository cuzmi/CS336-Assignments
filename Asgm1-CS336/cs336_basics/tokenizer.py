import regex as re
from typing import Iterator, Iterable

class GPT2Tokenizer:
    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] | None = None):
        self.vocab = vocab
        self.inv_vocab = {v: k for k, v in vocab.items()}
        self.bpe_ranks = {merge: i for i, merge in enumerate(merges)}
        
        self.special_tokens = special_tokens or []
        self.special_tokens_set = set(self.special_tokens)
        
        # 1. 编译普通的 GPT-2 正则表达式
        pat_str = r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        self.pat = re.compile(pat_str)
        
        # 2. 如果有特殊 Token，为其专门构建一个用于 re.split 的正则
        if self.special_tokens:
            # 关键修复 1：按照长度降序排序，确保长 Token (如 <|endoftext|><|endoftext|>) 优先匹配
            sorted_special = sorted(self.special_tokens, key=len, reverse=True)
            escaped_special = [re.escape(t) for t in sorted_special]
            # 使用捕获组 ()，这样 re.split 切分后，分隔符（即特殊 Token 本身）也会保留在列表中
            self.special_pat = re.compile("(" + "|".join(escaped_special) + ")")
        else:
            self.special_pat = None
            
        self.cache = {}

    def _bpe(self, token_bytes: bytes) -> list[bytes]:
        if token_bytes in self.cache:
            return self.cache[token_bytes]

        word = [bytes([b]) for b in token_bytes]
        if not word:
            return []

        while len(word) > 1:
            pairs = list(zip(word, word[1:]))
            best_pair = min(pairs, key=lambda p: self.bpe_ranks.get(p, float('inf')))
            
            if best_pair not in self.bpe_ranks:
                break

            new_word = []
            i = 0
            while i < len(word):
                if i < len(word) - 1 and word[i] == best_pair[0] and word[i+1] == best_pair[1]:
                    new_word.append(best_pair[0] + best_pair[1])
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            word = new_word

        self.cache[token_bytes] = word
        return word

    def encode(self, text: str) -> list[int]:
        bpe_tokens = []
        
        # 关键修复 2：先用特殊 Token 将文本切开
        if self.special_pat:
            chunks = self.special_pat.split(text)
        else:
            chunks = [text]
            
        for i, chunk in enumerate(chunks):
            if not chunk:
                continue
            
            # 因为 re.split 使用了捕获组，返回列表的奇数索引 (1, 3, 5...) 必定是我们匹配到的特殊 Token
            if self.special_pat and i % 2 == 1:
                bpe_tokens.append(chunk.encode("utf-8"))
            else:
                # 偶数索引是普通文本，走正常的 GPT-2 正则和 BPE 逻辑
                for match in self.pat.finditer(chunk):
                    token_str = match.group()
                    token_bytes = token_str.encode("utf-8")
                    bpe_tokens.extend(self._bpe(token_bytes))

        return [self.inv_vocab[b] for b in bpe_tokens]

    def decode(self, ids: list[int]) -> str:
        token_bytes = b"".join([self.vocab[i] for i in ids])
        return token_bytes.decode("utf-8", errors="replace")

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        # 同样的逻辑应用到流式处理中
        for line in iterable:
            if self.special_pat:
                chunks = self.special_pat.split(line)
            else:
                chunks = [line]

            for i, chunk in enumerate(chunks):
                if not chunk:
                    continue
                if self.special_pat and i % 2 == 1:
                    yield self.inv_vocab[chunk.encode("utf-8")]
                else:
                    for match in self.pat.finditer(chunk):
                        token_bytes = match.group().encode("utf-8")
                        for bpe_token in self._bpe(token_bytes):
                            yield self.inv_vocab[bpe_token]