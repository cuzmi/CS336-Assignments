"""
重构了ddp的 benchmark公共部分, 将特殊的all reduce gradient操作脱离出来
"""

import torch
import os

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

