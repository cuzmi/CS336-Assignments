import os
import random
import time
import numpy as np
from typing import Any, List, Tuple, Union
from collections import defaultdict

from tokenizer import Tokenizer

def sample_lines(
        file_path: Union[str, os.PathLike],
        sample_num: int
) -> List[str]:
    """Sample non-empty lines uniformly from a file using reservoir sampling."""
    rng = random.Random(42)
    reservoir: List[str] = []
    seen = 0

    with open(file_path, "r", encoding = "utf-8", errors = "ignore") as f:
        for line in f:
            line = line.rstrip('\n')
            if not line:
                continue
            seen += 1

            if len(reservoir) < sample_num:
                reservoir.append(line)
            else:
                j = rng.randint(0, seen - 1)
                if j < sample_num:
                    reservoir[j] = line
    
    if len(reservoir) < sample_num:
        raise ValueError(
            f"File contains fewer than {sample_num} non-empty samples."
        )
    
    return reservoir


def compression_ratio(
        cls,
        sample_num: int,
        tokenizer_file: Union[str, os.PathLike],
        samples: List[str],
        special_tokens: List[str],
        merge_file: Union[str, os.PathLike, None] = None,
        **kwargs
) -> Tuple[float, Any]:
    """Create a tokenizer and compute the average compression ratio on samples.

    Args:
        cls: Tokenizer class to instantiate.
        sample_num: Number of sampled examples.
        tokenizer_file: Path to the tokenizer pickle file.
        samples: Sampled text examples to encode.
        special_tokens: Special tokens passed to the tokenizer.
        merge_file: Optional separate merges file for older exports.
        **kwargs: Unused extra keyword arguments.

    Returns:
        A tuple of average bytes-per-token compression ratio and the tokenizer.
    """
    tokenizer = cls.from_files(
        str(tokenizer_file),
        str(merge_file or tokenizer_file),
        special_tokens,
    )

    samples_tokens = [len(tokenizer.encode(sample)) for sample in samples]
    samples_bytes = [len(sample.encode("utf-8")) for sample in samples]
    ratios = [byte_count / token_count for byte_count, token_count in zip(samples_bytes, samples_tokens) if token_count > 0]
    if not ratios:
        raise ValueError("No valid samples with non-zero token counts.")
    samples_ratio = sum(ratios) / len(ratios)

    return samples_ratio, tokenizer


def estimate_throughput(
        tokenizer: Tokenizer,
        samples: List[str],
) -> float:
    """Estimate tokenizer throughput in bytes per second on sampled text."""
    total_bytes = sum(len(sample.encode("utf-8")) for sample in samples)
    start_time = time.perf_counter()
    for sample in samples:
        tokenizer.encode(sample)
    elapsed = time.perf_counter() - start_time
    if elapsed <= 0:
        raise ValueError("Measured encoding time must be positive.")
    return total_bytes / elapsed


def test_main(
        cls,
        special_tokens: List[str],
        sample_num: int = 10,
        **kwargs,
):
    tns_path = kwargs['Tinystories']['original_path']
    tns_tokenizer_path = kwargs['Tinystories'].get('tokenizer_path', kwargs['Tinystories'].get('vocab_path'))
    tns_merge_path = kwargs['Tinystories'].get('merge_path')

    tns_samples = sample_lines(tns_path, sample_num)

    owt_path = kwargs['OpenWebText']['original_path']
    owt_tokenizer_path = kwargs['OpenWebText'].get('tokenizer_path', kwargs['OpenWebText'].get('vocab_path'))
    owt_merge_path = kwargs['OpenWebText'].get('merge_path')

    owt_samples = sample_lines(owt_path, sample_num)

    tns_result, tns_tokenizer = compression_ratio(
        cls,
        sample_num,
        tns_tokenizer_path,
        tns_samples,
        special_tokens,
        merge_file=tns_merge_path,
    )
    
    owt_result, owt_tokenizer = compression_ratio(
        cls,
        sample_num,
        owt_tokenizer_path,
        owt_samples,
        special_tokens,
        merge_file=owt_merge_path,
    )

    print("===== Assignment 1: compression ratio ======\n"
          f"tinystory: {tns_result:3f}\n"
          f"openwebtext: {owt_result:3f}\n")
    
    # ======== change tokenzier =========
    cross_result, _ = compression_ratio(
        cls,
        sample_num,
        tns_tokenizer_path,
        owt_samples,
        special_tokens,
        merge_file=tns_merge_path,
    )
    print("===== Assignment 2: Applying Tns tokenizer on owt sample ====\n"
          f"result: {cross_result:3f}")
    
    pile_size_bytes = 825 * 1024 * 1024 * 1024
    tns_throughput = estimate_throughput(tns_tokenizer, tns_samples)
    owt_throughput = estimate_throughput(owt_tokenizer, owt_samples)
    print("===== Assignment 3: Estimate tokenizer throughput ======\n"
          f"tinystory throughput: {tns_throughput:.3f} bytes/second\n"
          f"openwebtext throughput: {owt_throughput:.3f} bytes/second\n"
          f"tinystory time for 825GB: {pile_size_bytes / tns_throughput:.3f} seconds\n"
          f"openwebtext time for 825GB: {pile_size_bytes / owt_throughput:.3f} seconds\n")
    
    # ======= Record output ========= 
    tns_enc = []
    with open(tns_path, "r", encoding = "utf-8") as f:
        for line in f:
            tns_enc.extend(tns_tokenizer.encode(line))
    
    tns_arr = np.array(tns_enc, dtype = np.uint16)
    tns_output_dir = "./outputs/tns_encode.npy"

    owt_enc = []
    with open(owt_path, "r", encoding = "utf-8") as f:
        for line in f:
            owt_enc.extend(owt_tokenizer.encode(line))

    owt_arr = np.array(owt_enc, dtype = np.uint16)
    owt_output_dir = "./outputs/owt_encode.npy"
    
    os.makedirs("./outputs", exist_ok=True)
    np.save(tns_output_dir, tns_arr)
    np.save(owt_output_dir, owt_arr)
    print(f'Record Down!')


if __name__ == "__main__":
    special_tokens = ['<|endoftext|>']
    paths = defaultdict(str)
    paths = {
    "Tinystories": {
        "original_path": "./data/TinyStoriesV2-GPT4-valid.txt",
        "tokenizer_path": "./train/multi_bpe_tinystories.pkl",
    },
    "OpenWebText": {
        "original_path": "./data/owt_valid.txt",
        "tokenizer_path": "./train/multi_bpe_owt.pkl",
    },
    }

    
    test_main(Tokenizer, special_tokens, **paths) # 传入key， satisfy  **kwargs