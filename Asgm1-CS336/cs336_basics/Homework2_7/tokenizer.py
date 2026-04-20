import pickle
from typing import Iterable, Iterator

import regex as re


class Tokenizer:
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ) -> None:
        self.vocab = vocab
        self.inverse_vocab = {token_bytes: token_id for token_id, token_bytes in vocab.items()}
        self.merges = merges
        self.bpe_rank = {pair: rank for rank, pair in enumerate(merges)}

        self.special_tokens = set(special_tokens or [])
        if self.special_tokens:
            escaped = [re.escape(token) for token in sorted(self.special_tokens, key=len, reverse=True)]
            self.special_pat = re.compile("(" + "|".join(escaped) + ")")
        else:
            self.special_pat = None

        pat_str = r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        self.pat = re.compile(pat_str)
        self.cache: dict[bytes, list[bytes]] = {}

    @classmethod
    def from_files(
        cls,
        vocab_filepath: str,
        merges_filepath: str,
        special_tokens: list[str] | None = None,
    ) -> 'Tokenizer':
        with open(vocab_filepath, "rb") as vocab_file:
            vocab_data = pickle.load(vocab_file)

        if isinstance(vocab_data, dict) and "vocab" in vocab_data and "merges" in vocab_data:
            vocab = vocab_data["vocab"]
            merges = vocab_data["merges"]
        else:
            vocab = vocab_data
            with open(merges_filepath, "rb") as merges_file:
                merges = pickle.load(merges_file)

        return cls(vocab=vocab, merges=merges, special_tokens=special_tokens)

    def _bpe(self, token_bytes: bytes) -> list[bytes]:
        if token_bytes in self.cache:
            return self.cache[token_bytes]

        word = [bytes([byte]) for byte in token_bytes]
        if not word:
            return []

        while len(word) > 1:
            pairs = list(zip(word, word[1:]))
            best_pair = min(pairs, key=lambda pair: self.bpe_rank.get(pair, float("inf")))
            if best_pair not in self.bpe_rank:
                break

            merged_word = []
            i = 0
            while i < len(word):
                if i < len(word) - 1 and (word[i], word[i + 1]) == best_pair:
                    merged_word.append(best_pair[0] + best_pair[1])
                    i += 2
                else:
                    merged_word.append(word[i])
                    i += 1
            word = merged_word

        self.cache[token_bytes] = word
        return word

    def encode(self, text: str) -> list[int]:
        token_bytes_list: list[bytes] = []

        chunks = self.special_pat.split(text) if self.special_pat else [text]
        for index, chunk in enumerate(chunks):
            if not chunk:
                continue

            if self.special_pat and index % 2 == 1:
                token_bytes_list.append(chunk.encode("utf-8"))
                continue

            for match in self.pat.finditer(chunk):
                token_bytes_list.extend(self._bpe(match.group().encode("utf-8")))

        return [self.inverse_vocab[token_bytes] for token_bytes in token_bytes_list]

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for text in iterable:
            for token_id in self.encode(text):
                yield token_id

    def decode(self, ids: list[int]) -> str:
        token_bytes = b"".join(self.vocab[token_id] for token_id in ids)
        return token_bytes.decode("utf-8", errors="replace")


