"""
pre tokenization并行化, 得到corpus_freq后返回合并, 再在一个进程中运行bpe
"""
import os
import multiprocessing as mp
from typing import BinaryIO

import pickle
import regex
from collections import defaultdict
from typing import List, Tuple, Dict, Set, Union

import time
import tracemalloc

import train_bpe

def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))

def multi_run_train_bpe(
    token_frequency_table: Dict[Tuple[bytes, ...], int],
    vocab_size: int,
    special_tokens: List[str] | None = None,
    **kwargs,
) -> Tuple[Dict[int, bytes], List[Tuple[bytes, bytes]]]:
    """Given the path to an input corpus, run train a BPE tokenizer and
    output its vocabulary and merges.

    Args:
        vocab_size (int): Total number of items in the tokenizer's vocabulary (including special tokens).
        special_token (list[str]): A list of string speical tokens to be added to the tokenizer vocabulary.
            These strings will never be split into multiple tokens, and will always be
            kept as a single token. If these special tokens occur in the 'input_path',
            they are treated as any other string.
        token_freqnce_table (Dict[int]): A 

    Returns:
        tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
            vocab:
                The trained tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
                to bytes (token bytes)
            merges:
                BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
                representing that <token1> was merged with <token2>.
                Merges are ordered by order of creation.
    """
    if not isinstance(vocab_size, int) or vocab_size <= 0:
        raise ValueError("vocab_size 必须是一个正整数。")
        
    if special_tokens is None:
        special_tokens = []
       
    vocab: Dict[int, bytes] = {i: bytes([i]) for i in range(256)} 
    current_next_id: int = 256

    existing_byte_values: Set[bytes] = set(vocab.values())

    for st_str in special_tokens:
        if not st_str: 
            continue
        if len(vocab) >= vocab_size:
            break
        st_bytes = st_str.encode("utf-8")
        if st_bytes not in existing_byte_values: 
            vocab[current_next_id] = st_bytes
            existing_byte_values.add(st_bytes)
            current_next_id += 1

    # ====== 开始训练bpe =========
    merges: List[Tuple[bytes, bytes]] = []

    # 统计初始的所有 Token 对频率
    pair_counts = defaultdict(int)
    for token, freq in token_frequency_table.items():
        for i in range(len(token) - 1):
            pair_counts[token[i], token[i+1]] += freq

    # 4. BPE 合并主循环
    # NOTE: 核心 merge best pair -> affected token -> new token in dict, old token out / 维护了pair_count 和 token_freq - [By: Weijie] - 2026/03/13
    while len(vocab) < vocab_size:
        if not pair_counts:
            break

        # NOTE: 核心细节1 merge 找best pair的时候, 先频率，后pair本身 - [By: Weijie] - 2026/03/13
        best_pair = max(pair_counts.keys(), key=lambda p: (pair_counts[p], p))
        merges.append(best_pair)
        
        new_token_bytes = best_pair[0] + best_pair[1]
        vocab[current_next_id] = new_token_bytes
        current_next_id += 1
        
        bp0, bp1 = best_pair
        affected_tokens = []
        
        for token, freq in token_frequency_table.items():
            for i in range(len(token) - 1):
                if token[i] == bp0 and token[i+1] == bp1:
                    affected_tokens.append((token, freq))
                    break 
                    
        # 更新频率表
        for token, freq in affected_tokens:
            # 减去旧的 pair
            for i in range(len(token) - 1):
                pair = (token[i], token[i+1])
                pair_counts[pair] -= freq
                if pair_counts[pair] <= 0:
                    del pair_counts[pair]
                    
            # ---------- 原 merge_token_sequence 的内联逻辑 ----------
            new_seq = []
            i = 0
            seq_len = len(token)
            while i < seq_len:
                if i < seq_len - 1 and token[i] == bp0 and token[i+1] == bp1:
                    new_seq.append(new_token_bytes)
                    i += 2
                else:
                    new_seq.append(token[i])
                    i += 1
            new_token_frequency_seq = tuple(new_seq)
            # --------------------------------------------------------
            
            # 增加新的 pair
            for i in range(len(new_token_frequency_seq) - 1):
                pair = (new_token_frequency_seq[i], new_token_frequency_seq[i+1])
                pair_counts[pair] += freq
                
            del token_frequency_table[token]
            token_frequency_table[new_token_frequency_seq] += freq

    return vocab, merges

def pre_tokenization(input_path, start, end, special_tokens) -> Dict[Tuple[bytes, ...], int]:
    if special_tokens is None:
        special_tokens = []

    corpus_freq = defaultdict(int)  
 
    try:
        with open(input_path, "rb") as f:
            f.seek(start)
            text = f.read(end - start).decode("utf-8", errors="ignore")
    except FileNotFoundError:
        print(f"警告: 找不到文件 {input_path}，将作为空文本处理。")
        text = ""

    # ------------------ 正则切分 (Pre-tokenization) 开始 ------------------
    if special_tokens:
        pattern = '|'.join(map(regex.escape, special_tokens))
        chunks = regex.split(pattern, text)
    else:
        chunks = [text]

    PAT = regex.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
    
    for chunk in chunks:
        if not chunk:
            continue
        for match in PAT.finditer(chunk):
            word_bytes = match.group().encode("utf-8")
            bytes_tuple = tuple(bytes([x]) for x in word_bytes)
            corpus_freq[bytes_tuple] += 1
    
    return corpus_freq

def multiprocess_bpe_main(input_path, vocab_size, special_tokens):

    tracemalloc.start()
    start_time = time.perf_counter()
    ## Usage
    with open(input_path, "rb") as f:
        num_processes = 4
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")
        
    task = [
        (input_path, start, end, special_tokens)
        for start, end in zip(boundaries[:-1], boundaries[1:])
    ]

    with mp.Pool(num_processes) as pool:
        results = pool.starmap(pre_tokenization, task)

    # 汇总corpus_freq， 传入train bpe中
    corpus_table = defaultdict(int)
    for result in results:
        for corpus, freq in result.items():
            corpus_table[corpus] += freq

    vocab, merges = multi_run_train_bpe(corpus_table, vocab_size, special_tokens)

    end_time = time.perf_counter()
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    max_token = max(vocab.values(), key = len)

    output = {
        "vocab": vocab,
        "merges": merges
    }

    with open("multi_bpe_tinystories.pkl", "wb") as f:
        pickle.dump(output, f)

    print(f"multi_process_bpe:"
          f"total time: {end_time - start_time}"
          f"current memory: {current / 1024 /1024 :2f} MB"
          f"peak memory: {peak / 1024 / 1024 :2f} MB"
          f"max token: {max_token}, len: {len(max_token)}")

def singleprocess_bpe_main(input_path, vocab_size, special_tokens):
    tracemalloc.start()
    start_time = time.perf_counter()

    vocab, merges = train_bpe.run_train_bpe(input_path, vocab_size, special_tokens)

    end_time = time.perf_counter()
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    max_token = max(vocab.values(), key = len)

    output = {
        "vocab": vocab,
        "merges": merges
    }

    with open("single_bpe_tinystories.pkl", "wb") as f:
        pickle.dump(output, f)

    print(f"single_process_bpe:\n"
          f"total time: {end_time - start_time}\n"
          f"current memory: {current / 1024 /1024 :2f}MB\n"
          f"peak memory: {peak / 1024 / 1024 :2f}\nMB"
          f"max token: {max_token}, len: {len(max_token)}\n")

    
if __name__ == "__main__":
    input_path = "../data/TinyStoriesV2-GPT4-train.txt"
    vocab_size = 10000
    special_tokens = ['<|endoftext|>']

    singleprocess_bpe_main(input_path, vocab_size, special_tokens)
    multiprocess_bpe_main(input_path, vocab_size, special_tokens)

