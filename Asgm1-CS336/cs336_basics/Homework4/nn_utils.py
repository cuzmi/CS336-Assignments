"""
实现最后的组件
loss function + optimizer + training loop
"""

import torch
import numpy as np

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

def get_lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
):
    if it < warmup_iters:
        return (it / warmup_iters) * max_learning_rate
    
    if it < cosine_cycle_iters:
        angle = (it - warmup_iters) / (cosine_cycle_iters - warmup_iters)
        delta = max_learning_rate - min_learning_rate
        ratio = 0.5 * (1 + np.cos(angle * np.pi)) * delta

        return min_learning_rate + ratio
    
    return min_learning_rate