import numpy
import torch

def softmax(inputs: torch.Tensor, dim: int) -> torch.Tensor:
    max_logits = torch.max(inputs, dim = dim, keepdim=True).values
    exp_inputs = torch.exp(inputs - max_logits)
    dim_sum = torch.sum(exp_inputs, dim = dim, keepdim=True)

    return exp_inputs / dim_sum


def cross_entropy(inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    
    # ver 2 一切的妥协都是因为计算机的精度不足
    target_logits = torch.gather(inputs, dim = -1, index=targets.unsqueeze(-1))
    max_logits = torch.max(inputs, dim = -1, keepdim=True).values # 用来解决指数的上溢
    log_sum_exp = max_logits + torch.log(
        torch.sum(torch.exp(inputs - max_logits), dim = -1, keepdim=True)
        ) # 拆解公式解决了数值下溢
    loss = log_sum_exp - target_logits
    
    return loss.mean()

