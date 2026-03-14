import os
import pickle
import regex
from collections import defaultdict
from typing import List, Tuple, Dict, Set, Union

def run_train_bpe(
    input_path: Union[str, os.PathLike],
    vocab_size: int,
    special_tokens: List[str] | None = None,
    **kwargs,
) -> Tuple[Dict[int, bytes], List[Tuple[bytes, bytes]]]:
        # NOTE: 前置细节1 判断vocab_size是否合格 - [By: Weijie] - 2026/03/13
    if not isinstance(vocab_size, int) or vocab_size <= 0:
        raise ValueError("vocab_size 必须是一个正整数。")
        
    if special_tokens is None:
        special_tokens = []

    # 初始化基础词汇表 (0-255 字节)
        # TODO: 留意这些变量在后面的作用 - [By: Weijie] - 2026/03/13
    vocab: Dict[int, bytes] = {i: bytes([i]) for i in range(256)} # TODO: 核心 初始化 为什么要构建一个vocab词表呢，下面查看vocab的出现地方，分析用途 - [By: Weijie] - 2026/03/13
    current_next_id: int = 256

    token_frequency_table = defaultdict(int) # NOTE: 核心 初始化 预分词 的分组频率统计，为合并提供token和freq基础 - [By: Weijie] - 2026/03/13
    existing_byte_values: Set[bytes] = set(vocab.values()) # NOTE: 核心 初始化 为了记录所有的单字节bytes - [By: Weijie] - 2026/03/13

    # NOTE: 后置 初始化 特殊字符统计 - [By: Weijie] - 2026/03/13
    for st_str in special_tokens:
        if not st_str: 
            continue
        # NOTE: 后置细节1 初始化 vocab_size是固定好的，用于构建embedding hidden dim的构建 - [By: Weijie] - 2026/03/13
        if len(vocab) >= vocab_size:
            break
        st_bytes = st_str.encode("utf-8")
        if st_bytes not in existing_byte_values: 
            vocab[current_next_id] = st_bytes
            existing_byte_values.add(st_bytes)
            current_next_id += 1

    # 加载语料库       
    try:
        with open(input_path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
    except FileNotFoundError:
        print(f"警告: 找不到文件 {input_path}，将作为空文本处理。")
        text = ""

    # ------------------ 正则切分 (Pre-tokenization) 开始 ------------------
    # 第一步：用 special_tokens 进行大块切分，防止特殊字符被 BPE 破坏
    # NOTE: 前置 正则切分 特殊字符处理 - [By: Weijie] - 2026/03/13
    if special_tokens:
        pattern = '|'.join(map(regex.escape, special_tokens))
        # NOTE: 前置细节1 正则切分 排除特殊字符，不参与bpe合并 - [By: Weijie] - 2026/03/13
        chunks = regex.split(pattern, text)
    else:
        chunks = [text]

    # GPT-2 标准正则：匹配缩写、字母、数字、标点符号、连续空格等
    # 使用 \p{L} 和 \p{N} 需要 regex 库支持
    PAT = regex.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
    
    # 第二步：对文本块按“单词”级别进行切分
    # NOTE: 核心 正则切分， 得到token的频率 - [By: Weijie] - 2026/03/13
    for chunk in chunks:
        if not chunk:
            continue
        for match in PAT.finditer(chunk):
            word_bytes = match.group().encode("utf-8")
            bytes_tuple = tuple(bytes([x]) for x in word_bytes)
            token_frequency_table[bytes_tuple] += 1
    # ------------------ 正则切分 (Pre-tokenization) 结束 ------------------

    # NOTE: 前置 merge 统计所有pair的频率 - [By: Weijie] - 2026/03/13
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
        
        # 找出包含 best_pair 的 token 序列
        for token, freq in token_frequency_table.items():
            for i in range(len(token) - 1):
                if token[i] == bp0 and token[i+1] == bp1:
                    affected_tokens.append((token, freq))
                    break 
                    
        # 更新频率表
        # NOTE: 核心细节 merge 这里是选择token，减去整个token内部pair的count，而不是只减去affected 部分的前后 - [By: Weijie] - 2026/03/13
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
                
            # 更新总体频次字典
            del token_frequency_table[token]
            token_frequency_table[new_token_frequency_seq] += freq

    # 5. 保存文件
    with open("vocab.pkl", "wb") as f:
        pickle.dump(vocab, f)
    with open("merges.pkl", "wb") as f:
        pickle.dump(merges, f)

    return vocab, merges
