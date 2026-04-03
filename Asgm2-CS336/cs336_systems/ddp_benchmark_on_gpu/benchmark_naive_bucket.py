"""
1. measure the total time per train
2. measure the proportion of time spent on communicating gradients
三种方式测试的都是bakcward到step之间的额外通信成本
async / bucket就是wait的成本, naive ddp就是每次all reduce的成本, flat就是flat - all reduce - unflat的成本
"""

# build model -> measure gradients communication after backward, before step
import time
import os
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as f
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.optim import AdamW
from torch._utils import _unflatten_dense_tensors, _flatten_dense_tensors


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


handles = []

class LMBucketDDP(nn.Module):
    def __init__(self, module, bucket_size_mb):
        super().__init__()
        self.module = module
        self.world_size = dist.get_world_size()

        # broadcast & bucket
        with torch.no_grad():
            # NOTE: 比较有意思的点：各个节点都遍历p，然后src0的p能刚好broadcast到各个节点的p上 - [By: Weijie] - 2026/04/02
            for p in self.module.parameters():
                dist.broadcast(p, src = 0)
        
        self._set_bucket(bucket_size_mb)
        
        self.bucket_cnt = defaultdict(int)
        self.handles = []
        for p in self.module.parameters():
            if p.requires_grad:
                p.register_post_accumulate_grad_hook(self._backward_bucketed_hook)

    def _set_bucket(self, bucket_size_mb):
        
        self.buckets = []
        self.p_to_bucket_id = defaultdict(int)
        current_buckets = []
        current_buckets_bytes = 0
        current_idx = 0

        bucket_max_bytes = int(bucket_size_mb * 1024 * 1024)

        # backward: end to start / parameters: start to end 
        for p in reversed(list(self.module.parameters())):
            if not p.requires_grad:
                continue

            p_bytes = p.numel() * p.element_size()

            if current_buckets and current_buckets_bytes + p_bytes > bucket_max_bytes:
                self.buckets.append(current_buckets)

                current_buckets = []
                current_buckets_bytes = 0

                current_idx += 1

            current_buckets.append(p)
            current_buckets_bytes += p_bytes

            self.p_to_bucket_id[id(p)] = current_idx
        
        # 最后的bucket加入
        if current_buckets:
            self.buckets.append(current_buckets)

        return
    
    def _backward_bucketed_hook(self, param):
        # 放到init里面最好，(get world size放在里面会每次都要执行，放到synch只需要执行一次，但是会多处wait的时间) 
        # world_size = dist.get_world_size()
        # 判断是否运算
        bucket_id = self.p_to_bucket_id[id(param)]
        self.bucket_cnt[bucket_id] += 1
        if self.bucket_cnt[bucket_id] == len(self.buckets[bucket_id]):
            # flatten一次传输 or 遍历 para传输
            params = self.buckets[bucket_id]
            grads = [p.grad.div_(self.world_size) for p in params]
            flatten_tensor = _flatten_dense_tensors(grads)
            handle = dist.all_reduce(flatten_tensor, op = dist.ReduceOp.SUM, async_op = True)
            self.handles.append((handle, params, flatten_tensor))
            # unflatten能否在这里面执行？ - backward之后handle完成也在原地等待而没有立即执行， 还是存在时间冗余
    
    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)
    
    def finish_gradient_synchronization(self):
        for handle, params, flatten_tensor in self.handles:
            handle.wait()
            unflatten_tensor = _unflatten_dense_tensors(flatten_tensor, params)
            for grad, param in zip(unflatten_tensor, params):
                param.grad.copy_(grad)
        
        self.bucket_cnt.clear()
        self.handles.clear()


def train_one_step(LM, x_local, y_local, optimizer, device):
    """
    运行一次step 返回wait的时间差
    """
    optimizer.zero_grad(set_to_none = True)
 
    logits = LM(x_local) # (b, t, v)
    
    loss = f.cross_entropy(
        logits.reshape(-1, logits.shape[-1]), 
        y_local.reshape(-1))

    loss.backward() # 添加hook

    if torch.cuda.is_available():
        torch.cuda.synchronize(device)
    
    start = time.perf_counter()

    LM.finish_gradient_synchronization()

    if torch.cuda.is_available():
        torch.cuda.synchronize(device)
    
    end = time.perf_counter()

    optimizer.step()

    return end - start

def train_main(rank, world_size, bucket_size_mb, vocab_size, batch_size, context_length, warmup):
    device = setup(rank, world_size)

    x_local, y_local = get_train_batch(rank, world_size, batch_size, vocab_size, context_length, device)

    LM = model.BasicsTransformerLM(
        vocab_size, context_length, d_model, num_layers, num_heads, d_ff, rope_theta
    ).to(device)
    # NOTE: 前面to device，在通信前就放到了gpu上，更合理 - [By: Weijie] - 2026/04/02
    LMBucket = LMBucketDDP(LM, bucket_size_mb)

    optimizer = AdamW(LMBucket.parameters(), lr = 1e-4, betas = (0.99,0.999))

    for _ in range(warmup):
        _ = train_one_step(LMBucket, x_local, y_local, optimizer, device)

    for _ in range(5):
        if torch.cuda.is_available():
            torch.cuda.synchronize(device)
        
        start = time.perf_counter()

        wait_time = train_one_step(LMBucket, x_local, y_local, optimizer, device)

        if torch.cuda.is_available():
            torch.cuda.synchronize(device)

        end = time.perf_counter()

        train_time = end - start
        ratio = wait_time / train_time

        if rank == 0:
            print(
                f'[bucket_size: {bucket_size_mb}]'
                f'[rank: {rank}] '
                f'train_one_step = {train_time:.6f}s,'
                f'wait_time = {wait_time:.6f}s,'
                f'ratio = {ratio * 100:.3f}%'
            )

    dist.destroy_process_group()


if __name__ == '__main__':

    warmup = 5
    world_size = 2

    bucket_size_mbs = [1, 10, 100, 1000]
    for bucket_size_mb in bucket_size_mbs:
        cfg = (world_size, bucket_size_mb, vocab_size, batch_size, context_length, warmup)

        mp.spawn(train_main, cfg, nprocs = 2, join = True)