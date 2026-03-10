import torch
import torch.nn as nn
from torch.nn import functional as F


class AttentionHead(nn.Module):
    # normal attention head - k,q,v, head_size - B,T,C -> B,T,head_size
    def __init__(self, head_size):
        super().__init__()
        self.k = nn.Linear(n_embd, head_size)
        self.q = nn.Linear(n_embd, head_size)
        self.v = nn.Linear(n_embd, head_size)

        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):

        k = self.k(x) # B,T,head_size
        q = self.q(x)

        wei = q @ k.transpose(-1,-2) * k.shape[-1]**-0.5
        wei = F.softmax(wei, dim=-1) # B,T,T
        wei = self.dropout(wei)

        v = self.v(x)
        x = wei @ v

        return x # B,T,head_size

class MultiHead(nn.Module):
    # 多头 - cat - porj
    def __init__(self, head_nums):
        super().__init__()
        head_size = n_embd // head_nums # 假设可以被整除
        self.heads = nn.ModuleList([AttentionHead(head_size) for _ in range(head_nums)])
        self.proj = nn.Linear(n_embd, n_embd)

        self.dropout = nn.Dropout(dropout) # 可调节性，还是在内部单独设置一个dropout
    
    def forward(self, x):
        x = torch.cat([head(x) for head in self.heads], dim=-1)
        x = self.dropout(self.proj(x))

        return x # B,T,C
    
class FFwd(nn.Module):
    # 升维 - activation - 降维
    def __init__(self):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Linear(n_embd, 4*n_embd),
            nn.ReLU(),
            nn.Linear(4*n_embd, n_embd),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        x = self.seq(x)

        return x

class EncoderBlock(nn.Module):
    # 因为直接写Encoder要给出好多lyn1以及resident，所以直接复用block就好了
    def __init__(self, head_nums):
        super().__init__()
        self.mha = MultiHead(head_nums)
        self.ffwd = FFwd()
        
        self.lyn1 = nn.LayerNorm(n_embd)  # BatchNorm和LayerNorm的区别
        self.lyn2 = nn.LayerNorm(n_embd)
    
    def forward(self, x): # B,T,C
        x = x + self.mha(self.lyn1(x))
        x = x + self.ffwd(self.lyn2(x))

        return x

class Encoder(nn.Module):
    # n_layer, position, embedding, residental, Pre_Norm - 按照张量流程走一遍
    def __init__(self, n_layer, head_nums):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(block_size, n_embd)  # 这两个Embdding的维度确认????????

        self.seq = nn.Sequential(*[EncoderBlock(head_nums) for _ in range(n_layer)])
    
    def forward(self, x): # B,T
        B,T = x.shape

        x = self.emb(x) # B,T,C
        pos_tensor = torch.arange(0, T)
        x = x + self.pos_emb(pos_tensor)

        x = self.seq(x)

        return x # B,T,C

class MaskedAttentionHead(nn.Module):
    # normal attention head - k,q,v, head_size - B,T,C -> B,T,head_size
    def __init__(self, head_size):
        super().__init__()
        self.k = nn.Linear(n_embd, head_size)
        self.q = nn.Linear(n_embd, head_size)
        self.v = nn.Linear(n_embd, head_size)

        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size))) # 必须是方阵，为什么一定是block_size
    
    def forward(self, x):
        B,T,C = x.shape

        k = self.k(x) # B,T,head_size
        q = self.q(x)

        wei = q @ k.transpose(-1,-2) * k.shape[-1]**-0.5  # B,T,T
        wei = wei.masked_fill(self.tril[:T,:T] == 0, float('-inf'))  # 又是这个地方的写法不一致 masked_fill还是不清楚 / 中间是mask的条件
        wei = F.softmax(wei, dim=-1)

        v = self.v(x)
        x = wei @ v

        return x # B,T,head_size
    
class MaskedMultiHead(nn.Module):
    # 多头 - cat - porj
    def __init__(self, head_nums):
        super().__init__()
        head_size = n_embd // head_nums # 假设可以被整除
        self.heads = nn.ModuleList([MaskedAttentionHead(head_size) for _ in range(head_nums)])
        self.proj = nn.Linear(n_embd, n_embd)

        self.dropout = nn.Dropout(dropout) # 可调节性，还是在内部单独设置一个dropout
    
    def forward(self, x):
        x = torch.cat([head(x) for head in self.heads], dim=-1)
        x = self.dropout(self.proj(x))

        return x # B,T,C

class CrossAttentionHead(nn.Module):
    # 引入x_e
    def __init__(self, head_size):
        super().__init__()
        self.k = nn.Linear(n_embd, head_size)
        self.q = nn.Linear(n_embd, head_size)
        self.v = nn.Linear(n_embd, head_size)

        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x_e, x):

        k = self.k(x_e) # B,T,head_size
        q = self.q(x)

        wei = q @ k.transpose(-1,-2) * k.shape[-1]**-0.5
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)

        v = self.v(x_e)
        x = wei @ v

        return x # B,T,head_size

class CrossMultiHead(nn.Module):
    # 多头 - cat - porj
    def __init__(self, head_nums):
        super().__init__()
        head_size = n_embd // head_nums # 假设可以被整除
        self.heads = nn.ModuleList([CrossAttentionHead(head_size) for _ in range(head_nums)])
        self.proj = nn.Linear(n_embd, n_embd)

        self.dropout = nn.Dropout(dropout) # 可调节性，还是在内部单独设置一个dropout
    
    def forward(self, x_e, x):
        x = torch.cat([head(x_e, x) for head in self.heads], dim=-1)
        x = self.dropout(self.proj(x))

        return x # B,T,C

class DecoderBlock(nn.Module):
    # Pre - Mask - Res - Pre - Cross - Res - Pre - FFwd - Res
    def __init__(self, head_nums):
        super().__init__()
        self.mmha = MaskedMultiHead(head_nums)
        self.cmha = CrossMultiHead(head_nums)
        self.ffwd = FFwd()

        self.lyn1 = nn.LayerNorm(n_embd)
        self.lyn2 = nn.LayerNorm(n_embd)
        self.lyn3 = nn.LayerNorm(n_embd)
    
    def forward(self, x_e, x):
        x = x + self.mmha(self.lyn1(x))
        x = x + self.cmha(x_e, self.lyn2(x))
        x = x + self.ffwd(self.lyn3(x))

        return x

class Decoder(nn.Module):
    # x - Block - out
    def __init__(self, n_layer, head_nums):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(block_size, n_embd)
        self.blocks = nn.ModuleList([DecoderBlock(head_nums) for _ in range(n_layer)])

        self.ln = nn.Linear(n_embd, vocab_size)
    
    def forward(self, x_e, x): #B,T
        B,T = x.shape

        x = self.embed(x)
        pos_emb = self.pos_emb(torch.arange(0,T))
        x = x + pos_emb

        for block in self.blocks:
            x = block(x_e, x)

        x = self.ln(x) # B,T,vocab_size

        return x

    





