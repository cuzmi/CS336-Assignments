"""
add data_loader, checkpoint, training loop
"""
import torch
import numpy as np
from typing import Tuple

def get_batch(dataset, batch_size, context_length, device) -> Tuple[torch.Tensor, ...]:
    # 检查devcie 是否存在
    if torch.cuda.is_available():
        raise NotImplemented
    
    indices = torch.randint(0, len(dataset) - context_length, size = (batch_size, )).tolist()
    x = torch.stack([torch.tensor(dataset[i: i+ context_length], dtype = torch.long) for i in indices], dim = 0).to(device)
    y = torch.stack([torch.tensor(dataset[i + 1: i + context_length + 1], dtype = torch.long) for i in indices], dim = 0).to(device)

    return (x, y)