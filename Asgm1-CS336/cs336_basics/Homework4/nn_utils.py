"""
实现最后的组件
loss function + optimizer + training loop
"""

import torch

def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    # 有可能产生shape变化的数值操作要留意是否保持 shape
    dim_max = torch.max(x, dim = dim, keepdim = True).values
    exp = torch.exp(x - dim_max)
    exp_sum = torch.sum(exp, dim = dim, keepdim = True)
    return  exp / exp_sum

def cross_entropy(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    # 容易产生下溢问题， 先算prob 则容易造成 log -> - ∞
    # x = softmax(x, dim = -1)

    target_logits = torch.gather(x, dim = -1, index = y.unsqueeze(-1)) # b,t,1
    max_logits = torch.max(x, dim = -1, keepdim = True).values
    log_sum_exp = max_logits + torch.log(
        torch.sum(torch.exp(x - max_logits), dim = -1, keepdim = True)
    )
    loss = log_sum_exp - target_logits
    return loss.mean()