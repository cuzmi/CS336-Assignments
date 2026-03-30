"""
1. measure the total time per train
2. measure the proportion of time spent on communicating gradients
"""

# build model -> measure gradients communication after backward, before step
import torch
import time
import os

import torch.nn.functional as f
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.optim import AdamW

from cs336_basics import model

vocab_size = 10000
d_model = 1600
d_ff = 6400
num_layers = 48
num_heads = 25
context_length = 256
rope_theta = 10000.0
batch_size = 2


handles = []
# dataset - ddp - x, y 工程化Dataset 在原理阶段没有必要实现
# class RandomTokenDataset(Dataset):
#     def __init__(self, num_samples, context_length, vocab_size):
#         super().__init__()
#         self.num_samples = num_samples
#         self.context_length = context_length
#         self.vocab_szie = vocab_size
    
#     def __len__(self):
#         return self.num_samples
    
#     def __getitem__(self, index):

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12900'
    if torch.cuda.is_available():
        torch.cuda.set_device(rank)
        dist.init_process_group(
            backend = 'nccl',
            world_size = world_size,
            rank = rank
        )
        device = torch.device(f'cuda:{rank}')
    else:
        dist.init_process_group(
            backend = 'gloo',
            world_size = world_size,
            rank = rank
        )
        device = torch.device('cpu')
    
    return device

def backward_all_reduce(param):
    global handles
    if param.grad is None:
        return 
    handle = dist.all_reduce(param.grad, op = dist.ReduceOp.SUM, async_op = True)
    handles.append((param,handle))
    

# TODO: 回顾一下LLM的dataset shape - [By: Weijie] - 2026/03/26
def get_train_batch(rank, world_size, vocab_size, batch_size, context_length, device):
    """
    每一部分预留位置 - 在rank = 0 生成数据 - scatter到各个位置
    """
    assert batch_size % world_size == 0
    local_bs = batch_size // world_size

    x_local = torch.empty(
        (local_bs, context_length),
        device = device,
        dtype = torch.long
    )
    y_local = torch.empty(
        (local_bs, context_length),
        device = device,
        dtype = torch.long
    )

    if rank == 0:
        torch.manual_seed(42)

        global_x = torch.randint(0, vocab_size, (batch_size, context_length), dtype = torch.long, device = device)
        global_y = torch.randint(0, vocab_size, (batch_size, context_length), dtype = torch.long, device = device)


        x_chunk = list(global_x.chunk(world_size, dim = 0))
        y_chunk = list(global_y.chunk(world_size, dim = 0))
    else:
        x_chunk = None
        y_chunk = None

    dist.scatter(x_local, scatter_list = x_chunk, src = 0)
    dist.scatter(y_local, scatter_list = y_chunk, src = 0)

    return x_local, y_local

def train_one_step(LM, x, y, optimizer, device, world_size) -> float: 
    # NOTE: optimzier 清空模型上的grad - [By: Weijie] - 2026/03/27
    optimizer.zero_grad(set_to_none = True)

    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    
    handles.clear()
    # x - local_bs, context_length; y, local_bs, context_length； logits local_bs, context_length, vocab_size
    logits = LM(x)
    loss = f.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        y.reshape(-1)
    )

    # benchmarking
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    ar_start = time.perf_counter()

    loss.backward()

    for param, handle in handles:
        handle.wait()
        param.grad.div_(world_size)
    
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    ar_end = time.perf_counter()
    
    optimizer.step()

    return ar_end - ar_start

# train_main 是用来做作为主内容训练的
# setup - batch 生成分配 + model + optimzier 建立 - 训练 - grad reduce
def train_main(rank, world_size, vocab_size, batch_size, context_length, warmup):
    device = setup(rank, world_size)

    # 生成数据集 ~ 每个进程都分配到一份
    x, y = get_train_batch(rank, world_size, vocab_size, batch_size, context_length, device)

    LM = model.BasicsTransformerLM(
        vocab_size,
        context_length,
        d_model,
        num_layers,
        num_heads,
        d_ff,
        rope_theta,
    ).to(device)

    for p in LM.parameters():
        if p.requires_grad:
            p.register_post_accumulate_grad_hook(backward_all_reduce)

    # NOTE: optimzier本身不是数据，不是包含很多内容的对象，而是一种作用方法，不用to device - [By: Weijie] - 2026/03/27
    optimizer = AdamW(params = LM.parameters(), lr = 1e-4, betas = (0.99,0.999))

    LM.train()
    # NOTE: warmup就是走几遍要记时的全流程 - [By: Weijie] - 2026/03/27
    for _ in range(warmup):
        _ = train_one_step(LM, x, y, optimizer, device, world_size)

    # 正式训练和benchmark
    for _ in range(2):
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
        start = time.perf_counter()

        all_reduce_time = train_one_step(LM, x, y, optimizer, device, world_size)
        
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
        end = time.perf_counter()

        train_time = end - start
        ratio = all_reduce_time / train_time

        if rank == 0:
            print(
                f"[rank {rank}] "
                f"train_one_step={train_time:.6f}s, "
                f"all_reduce_time={all_reduce_time:.6f}s, "
                f"ratio={ratio * 100:.3f}%"
            )

    dist.destroy_process_group()

if __name__ == "__main__":

    world_size = 2
    warmup = 2

    cfg = (world_size, vocab_size, batch_size, context_length, warmup)

    mp.spawn(fn = train_main, args = cfg, nprocs = world_size, join = True)
    
