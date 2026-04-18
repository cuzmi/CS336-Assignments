"""
创建很多的torch layer 部分, 通过test, 之后用于构建一整个transformer_lm
"""
import torch
import torch.nn as nn
from jaxtyping import Int, Float

# 工程上采用的不是课本上的 W@x 运算, 而是 x@W, 这样方便处理batch 所以W 一般存储为 (d_out, d_in)
# e.g. (batch, d_in) @ (d_out, d_in).T  如果还是W在前面，那么就会要多纬度变化了
"""
Linear:
1. 注意W.shape = [d_out, d_in]
2. class 内部可以不在init 创建实例
3. 更改权重不要轻易被计入计算图， 不然会对更改权重这个操作计算梯度 no grad
"""
class Linear(nn.Module):
    def __init__(self, in_features, out_features, device = None, dtype = None):
        # super不是创建实例，而是告诉接下来的实例汇集成父类的方法
        super().__init__()

        self.W = nn.Parameter(
            data = torch.empty(out_features, in_features, device = device, dtype = dtype),
            requires_grad = True
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = x @ self.W.T

        return output
    
"""
Embedding:
1. 高级索引的操作, 很精妙
"""
class Embedding(nn.Module):
    def __init__(self, vocab_size, d_model, device = None, dtype = None):
        super().__init__()

        self.W = nn.Parameter(
            data = torch.empty(vocab_size, d_model, device = device, dtype = dtype),
            requires_grad = True
        )

    def forward(self, x: Int[torch.Tensor, "..."]):
        # 这里不是矩阵乘法, 用高级索引可以直接把x的每个元素用于获取w的dim=0的所有元素
        return self.W[x]

"""
softmax:
1. 注意针对每个元素的操作， 和以前的以矩阵为基本单位不同， 这里是对每个元素的操作
"""
def softmax(in_features, dim) -> Float[torch.Tensor, "..."]:
    max_logits = torch.max(in_features, dim = dim, keepdim = True).values
    exp_inp = torch.exp(in_features - max_logits)
    dim_sum = torch.sum(exp_inp, dim = dim, keepdim = True)

    return exp_inp / dim_sum


class CausalMultiHeadAttention(nn.Module):
    # 1. scaled_dot 自己实现mask 2. masked fill 另一种mask方法
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_size = d_model // self.num_heads
        self.Q = torch.empty((d_model, d_model))
        self.K = torch.empty((d_model, d_model))
        self.V = torch.empty((d_model, d_model))

        self.sm = nn.Softmax(dim = -1)

        self.W_o = torch.empty((d_model, d_model))
    
    # 在forward 输出 B,N,N
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape

        head_outputs = []

        for i in range(self.num_heads):
            q = self.Q[i*self.head_size:(i+1) * self.head_size, :]
            k = self.K[i*self.head_size:(i+1) * self.head_size, :]
            v = self.V[i*self.head_size:(i+1) * self.head_size, :]


            q = x @ q.T
            k = x @ k.T
            v = x @ v.T

            scores = ( q @ k.transpose(-2,-1) ) * (self.head_size ** -0.5) # 使用transpose而不是T
            # mask
            mask = torch.triu(torch.ones(T, T, dtype = torch.bool), diagonal = 1)
            scores = scores.masked_fill(mask, float("-inf"))

            scores = self.sm(scores)
            
            head_output = scores @ v # 无x存在
            head_outputs.append(head_output)
        
        output = torch.cat(head_outputs, dim = -1)
        output = output @ self.W_o.T

        return output