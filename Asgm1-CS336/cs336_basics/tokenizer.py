import regex as re
from typing import Iterator, Iterable, List

class GPT2Tokenizer:
    def __init__(self, vocab: dict[int, bytes], merge: List[tuple[bytes, bytes]], special_tokens: List[str] | None = None):
        # 初始化操作
        self.vocab = vocab # NOTE: decode base - [By: Weijie] - 2026/03/12
        self.enc_vocab = {b: i for i, b in vocab.items()}
        self.bpe_rank = {m:i for i, m in enumerate(merge)}

        self.special_tokens = special_tokens or []
        self.special_tokens = set(self.special_tokens)

            # NOTE: 特殊token判断 - [By: Weijie] - 2026/03/12
        if special_tokens:
            sorted_speical = sorted(self.special_tokens, key=len, reverse=True)
            escape_special = [re.escape(t) for t in sorted_speical]
                # TODO: 正则表达式的通用形式 1. 构建special用到了 2. common的pet str理解 - [By: Weijie] - 2026/03/12
            self.special_pet = re.compile('(' + '|'.join(escape_special) + ')')
        else:
            self.special_pet = None

        pat_str = r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        self.pet = re.compile(pat_str)

        self.cache = {}

    def _bpe(self, token_bytes):
        # NOTE: 核心  匹配 pair 并用新word来覆盖替代 - [By: Weijie] - 2026/03/12

            # NOTE: 前置细节1 判断是否存在cache - [By: Weijie] - 2026/03/12
        if token_bytes in self.cache:
            return self.cache[token_bytes]
            # NOTE: 前置细节2 进行bytes转换， 将提取的int转化为对应的bytes - [By: Weijie] - 2026/03/12
        word = [bytes([b]) for b in token_bytes]

        if not word:
            return []
        
        while len(word) > 1:
            pairs = list(zip(word, word[1:]))
            best_pair = min(pairs, key= lambda p: self.bpe_rank.get(p, float('inf'))) # NOTE: lambda 本身不遍历，它只是定义了“对每个元素计算值的方法”。在这里实现遍历的是min - [By: Weijie] - 2026/03/12
            # NOTE: 内部细节1 pair不能存在直接break - [By: Weijie] - 2026/03/12
            if best_pair not in self.bpe_rank:
                break

            new_word = []
            i = 0
            while i <len(word):
                if i < len(word) - 1 and (word[i], word[i+1]) == best_pair:
                    new_word.append(best_pair[0] + best_pair[1])
                    i += 2
                else:
                     new_word.append(word[i])
                     i += 1
            word = new_word

        return word
    
    def encode(self, text: str) -> List[int]:
        # NOTE: 核心： 先匹配speical在匹配common - [By: Weijie] - 2026/03/12

            # NOTE: 前置细节1. 判断special是否存在 - [By: Weijie] - 2026/03/12
        bpe_tokens = []

        if self.special_pet:
            chunks = self.special_pet.split(text)
        else:
            chunks = [text]

        for i, chunk in enumerate(chunks):
            # NOTE: 内置细节1. 特殊切分会导致空字符 - [By: Weijie] - 2026/03/12
            if not chunk:
                continue

            if self.special_pet and i % 2 == 1:
                bpe_tokens.append(chunk.encode('utf-8'))
            else:
                for match in self.pet.finditer(chunk):
                    token_str = match.group()
                    token_bytes = token_str.encode('utf-8')
                    bpe_tokens.extend(self._bpe(token_bytes)) # NOTE: 内置细节2 extend而不是append，解包List - [By: Weijie] - 2026/03/12
        
        return [self.enc_vocab[token] for token in bpe_tokens]

    def decode(self, ids: List[int]) -> str:
        # NOTE: 核心 解码为str - [By: Weijie] - 2026/03/12
            # NOTE: 前置细节1 将int转换为bytes，方便后面decode - [By: Weijie] - 2026/03/12
        token_bytes = b"".join([self.vocab[i] for i in ids])
        return token_bytes.decode('utf-8', errors='replace') # NOTE: 内置细节1 无法解码单独的非完整的utf-8字符，会产生报错 - [By: Weijie] - 2026/03/12

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        # NOTE: 核心 流式输出encode - [By: Weijie] - 2026/03/12
            # NOTE: 前置细节1 流式读取，line / 和encode基本相似，但是存在一些用法上的细节 - [By: Weijie] - 2026/03/12
        for line in iterable:
            if self.special_tokens:
                chunks = self.special_pet.split(line)
            else:
                chunks = [line]

            for i, chunk in enumerate(chunks):
                if not chunk:
                    continue

                if self.special_pet and i % 2 == 1:
                    yield self.enc_vocab[chunk.encode('utf-8')]
                else:
                    for match in self.pet.finditer(chunk):
                        token_str = match.group()
                        token_bytes = token_str.encode('utf-8')
                        for bpe_token in self._bpe(token_bytes):
                            yield self.enc_vocab[bpe_token]
