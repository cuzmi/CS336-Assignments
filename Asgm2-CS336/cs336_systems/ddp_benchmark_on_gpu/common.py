"""
重构了ddp的 benchmark公共部分, 将特殊的all reduce gradient操作脱离出来
"""

import torch
import os
import sys
import subprocess

import torch.distributed as dist


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = "localhost"
    os.environ['MASTER_PORT'] = "12900"
    if torch.cuda.is_available():
        torch.cuda.set_device(rank)
        dist.init_process_group(
            backend = "nccl",
            rank = rank,
            world_size = world_size
        )
        device = torch.device(f'cuda:{rank}')
    else:
        dist.init_process_group(
            backend = 'gloo',
            rank = rank,
            world_size = world_size
        )
        device = torch.device('cpu')
    
    return device

def get_train_batch(rank, world_size, batch_size, vocab_size, context_length, device):
    """
    rank上预留空间, rank = 0 生成global, 切分到rank的local上
    """
    assert batch_size % world_size == 0
    local_bs = batch_size // world_size

    x_local = torch.empty(
        (local_bs, context_length), device = device, dtype = torch.long
    )
    y_local = torch.empty(
        (local_bs, context_length), device = device, dtype = torch.long
    )

    if rank == 0:
        torch.manual_seed(42)

        x_global = torch.randint(0, vocab_size, (batch_size, context_length), dtype = torch.long, device = device)
        y_global = torch.randint(0, vocab_size, (batch_size, context_length), dtype = torch.long, device = device)

        x_chunks = list(x_global.chunk(world_size, dim = 0))
        y_chunks = list(y_global.chunk(world_size, dim = 0))
    else:
        x_chunks = None
        y_chunks = None

    # NOTE: 按照rank排列好对应的tensor_list， 是按照rank i 分配list i - [By: Weijie] - 2026/03/30
    dist.scatter(x_local, x_chunks, src = 0)
    dist.scatter(y_local, y_chunks, src = 0)

    return x_local, y_local


def report_rank_metrics(rank, world_size, title, metrics):
    """
    Gather scalar metrics from all ranks and print per-rank values plus
    the max across ranks on rank 0.
    """
    gathered = [None for _ in range(world_size)] if rank == 0 else None
    dist.gather_object(metrics, object_gather_list = gathered, dst = 0)

    if rank != 0:
        return

    print(f"\n=== {title} ===")
    metric_names = list(metrics.keys())
    for metric_name in metric_names:
        max_value = float("-inf")
        for r, rank_metrics in enumerate(gathered):
            value = rank_metrics[metric_name]
            max_value = max(max_value, value)
            if "ratio_pct" in metric_name:
                print(f"rank {r}: {metric_name}={value:.3f}%")
            else:
                print(f"rank {r}: {metric_name}={value:.6f}s")
        if "ratio_pct" in metric_name:
            print(f"max_across_ranks: {metric_name}={max_value:.3f}%")
        else:
            print(f"max_across_ranks: {metric_name}={max_value:.6f}s")


def relaunch_with_nsys_if_requested(module_name, output_stem):
    """
    Relaunch the current module under Nsight Systems when AUTO_NSYS=1.
    This lets the user run the benchmark entrypoint directly while still
    producing a .nsys-rep file.
    """
    if os.environ.get("AUTO_NSYS") != "1":
        return
    if os.environ.get("NSYS_ACTIVE") == "1":
        return

    nsys_exe = os.environ.get("NSYS_EXE", "nsys")
    python_exe = sys.executable
    output_dir = os.environ.get("NSYS_OUTPUT_DIR", "nsys_reports")
    os.makedirs(output_dir, exist_ok = True)

    child_env = os.environ.copy()
    child_env["NSYS_ACTIVE"] = "1"

    command = [
        nsys_exe,
        "profile",
        "--trace",
        "cuda,nvtx,osrt,cudnn,cublas",
        "--sample",
        "none",
        "--force-overwrite",
        "true",
        "--output",
        os.path.join(output_dir, output_stem),
        python_exe,
        "-m",
        module_name,
    ]

    subprocess.run(command, check = True, env = child_env)
    raise SystemExit(0)

