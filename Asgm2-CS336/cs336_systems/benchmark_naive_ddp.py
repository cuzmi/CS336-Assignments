"""
1. measure the total time per train
2. measure the proportion of time spent on communicating gradients
"""

# build model -> measure gradients communication after backward, before step
import torch
import time
import os
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.optim import AdamW

from torch.utils.data import Dataset, DataLoader
from cs336_basics import model


world_size = 2
device_type = 'gpu'



vocab_size = 10000
d_model = 1600
d_ff = 6400
num_layers = 48
num_heads = 25
context_length = 256
rope_theta = 10000.0
batch_size = 64

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

# TODO: 回顾一下LLM的dataset shape - [By: Weijie] - 2026/03/26
def get_train_batch(rank, world_size, vocab_size, batch_size, context_length, device):
    """
    每一部分预留位置 - 在rank = 0 生成数据 - scatter到各个位置
    """
    assert batch_size % world_size == 0
    local_bs = batch_size // world_size

    x_local = torch.empty(
        (local_bs, context_length),
        device = device, # TODO: 这里的device = device不是多余吗，因为本来就在当前rank上使用empty生成 - [By: Weijie] - 2026/03/26
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

# train_main 是用来做作为主内容训练的
# setup - batch 生成分配 + model + optimzier 建立 - 训练 - grad reduce
def train_main(rank, world_size, vocab_size):
    device = setup(rank, world_size)

    # 生成数据集 ~ 可以一起实现
    x, y = get_train_batch()


    LM = model.BasicsTransformerLM(
        vocab_size,
        context_length,
        d_model,
        num_layers,
        num_heads,
        d_ff,
        rope_theta
    )

    optimizer = AdamW(params = LM.parameters, lr = 1e-4, betas = [0.99,0.999])

    # 单步训练

    return

if __name__ == "__main__":

    mp.spawn(fn = train_main, args = (world_size, device_type), npcrocs = world_size, join = True)
    


    # ~ warmup + sy -> for train measurement

    # forward 


    # backward - ddp
