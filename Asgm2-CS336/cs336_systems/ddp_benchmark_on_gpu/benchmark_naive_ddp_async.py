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


class OverlapDDP(torch.nn.Module):
    def __init__(self, module):
        """
        grad + hook; synchronsize for handle wait; step = setp
        """
        # NOTE: module 存在检查attr的属性， module/para等要先创建好实例才能self.module - [By: Weijie] - 2026/04/04
        super().__init__()
        self.module = module

        self.handles = []

        for p in self.module.parameters():
            dist.broadcast(p, src = 0)
        
        for p in self.module.parameters():
            if p.requires_grad:
                p.register_post_accumulate_grad_hook(self.backward_hook)
    
    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    def backward_hook(self, param):
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()

        if param.grad is None:
            return
        handle = dist.all_reduce(param.grad, dist.ReduceOp.SUM, async_op = True)
        self.handles.append((param,handle))

    def finish_gradient_synchronization(self):
        for param, handle in self.handles:
            handle.wait()
            param.grad.div_(self.world_size)

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

    # benchmarking
    loss.backward()

    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    ar_start = time.perf_counter()

    LM.finish_gradient_synchronization()
    
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
    x, y = get_train_batch(rank, world_size, batch_size, vocab_size, context_length, device)

    LM = model.BasicsTransformerLM(
        vocab_size,
        context_length,
        d_model,
        num_layers,
        num_heads,
        d_ff,
        rope_theta,
    ).to(device)

    LM = OverlapDDP(LM)

    # NOTE: optimzier本身不是数据，不是包含很多内容的对象，而是一种作用方法，不用to device - [By: Weijie] - 2026/03/27
    optimizer = AdamW(params = LM.parameters(), lr = 1e-4, betas = (0.99,0.999))

    LM.train()
    # NOTE: warmup就是走几遍要记时的全流程 - [By: Weijie] - 2026/03/27
    for _ in range(warmup):
        _ = train_one_step(LM, x, y, optimizer, device, world_size)

    # 正式训练和benchmark
    for _ in range(5):
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
        start = time.perf_counter()

        sync_time = train_one_step(LM, x, y, optimizer, device, world_size)
        
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
        end = time.perf_counter()

        train_time = end - start
        ratio = sync_time / train_time

        if rank == 0:
            print(
                f"[rank {rank}] "
                f"train_one_step={train_time:.6f}s, "
                f"sync_time={sync_time:.6f}s, "
                f"ratio={ratio * 100:.3f}%"
            )

    dist.destroy_process_group()

if __name__ == "__main__":

    world_size = 2
    warmup = 5

    cfg = (world_size, vocab_size, batch_size, context_length, warmup)

    mp.spawn(fn = train_main, args = cfg, nprocs = world_size, join = True)