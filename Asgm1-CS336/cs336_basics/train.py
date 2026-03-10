import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.optim import AdamW

# 导入你的模块
from BPETokenizer import BPETokenizer
from Transformer import Decoder

# B,T,C
batch_size = 4
block_size = 32  # 序列长度
max_iters = 1000
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# get batch
def get_batch(data): # data is original data
    ix = torch.randint(len(data) - block_size, (batch_size,)) # [start1, start2, ...]
    x = torch.stack([data[start: start + block_size] for start in ix])
    y = torch.stack([data[start+1:start+1+block_size] for start in ix])

    return x.to(device), y.to(device)