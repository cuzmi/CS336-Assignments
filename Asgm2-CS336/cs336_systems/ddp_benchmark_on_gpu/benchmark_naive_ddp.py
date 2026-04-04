"""
1. measure the total time per train
2. measure the proportion of time spent on communicating gradients
三种方式测试的都是bakcward到step之间的额外通信成本
async / bucket就是wait的成本, naive ddp就是每次all reduce的成本, flat就是flat - all reduce - unflat的成本
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
from .common import setup, get_train_batch

vocab_size = 10000
d_model = 1600
d_ff = 6400
num_layers = 48
num_heads = 25
context_length = 256
rope_theta = 10000.0
batch_size = 2

vocab_size = 1024
d_model = 128
d_ff = 512
num_layers = 2
num_heads = 4
context_length = 32
rope_theta = 10000.0
batch_size = 1



def train_one_step(LM, x, y, optimizer, device, world_size) -> float: 
    # NOTE: optimzier 清空模型上的grad - [By: Weijie] - 2026/03/27
    optimizer.zero_grad(set_to_none = True)

    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    
    # x - local_bs, context_length; y, local_bs, context_length； logits local_bs, context_length, vocab_size
    logits = LM(x)
    loss = f.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        y.reshape(-1)
    )
    loss.backward()

    # benchmarking
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    ar_start = time.perf_counter()

    for p in LM.parameters():
        if p.grad is None:
            continue
        dist.all_reduce(p.grad, op = dist.ReduceOp.SUM)
        p.grad.div_(world_size)
    
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

    with torch.no_grad():
        for p in LM.parameters():
            dist.broadcast(p, src = 0)



    # NOTE: optimzier本身不是数据，不是包含很多内容的对象，而是一种作用方法，不用to device - [By: Weijie] - 2026/03/27
    optimizer = AdamW(params = LM.parameters(), lr = 1e-4, betas = (0.99,0.999))

    LM.train()
    train_times = []
    all_reduce_times = []
    # NOTE: warmup就是走几遍要记时的全流程 - [By: Weijie] - 2026/03/27
    for _ in range(warmup):
        _ = train_one_step(LM, x, y, optimizer, device, world_size)

    # 正式训练和benchmark
    for _ in range(5):
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
        start = time.perf_counter()

        all_train_time = train_one_step(LM, x, y, optimizer, device, world_size)
        
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
        end = time.perf_counter()

        train_time = end - start
        ratio = all_train_time / train_time
        train_times.append(train_time)
        all_reduce_times.append(all_train_time)

        if rank == 0:
            print(
                f"[rank {rank}] "
                f"train_one_step={train_time:.6f}s, "
                f"all_reduce_time={all_train_time:.6f}s, "
                f"ratio={ratio * 100:.3f}%"
            )

    avg_train_time = sum(train_times) / len(train_times)
    avg_all_reduce_time = sum(all_reduce_times) / len(all_reduce_times)
    avg_ratio = avg_all_reduce_time / avg_train_time

    if rank == 0:
        print(
            f"[rank {rank}] "
            f"avg_train_one_step={avg_train_time:.6f}s, "
            f"avg_all_reduce_time={avg_all_reduce_time:.6f}s, "
            f"avg_ratio={avg_ratio * 100:.3f}%"
        )

    dist.destroy_process_group()

if __name__ == "__main__":

    world_size = 2
    warmup = 5

    cfg = (world_size, vocab_size, batch_size, context_length, warmup)

    mp.spawn(fn = train_main, args = cfg, nprocs = world_size, join = True)
    
