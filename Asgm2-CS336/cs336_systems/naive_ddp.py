"""
backward pass - all reduce == single process training outcome
"""

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp

class DDPIndividualParameters(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module
        # NOTE: broadcast操作, src = 0 传播p.data ,src != 0 ， 将本地的的p.data 由传播的覆盖 - [By: Weijie] - 2026/03/26
        # NOTE: 如果想改变映射对象，则可以取中间变量桥接 eg: A <- tensor = tensor -> B - [By: Weijie] - 2026/03/26
        for p in self.module.parameters():
            dist.broadcast(p.data, src = 0)
    
    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    # NOTE: 同类型数据可以直接调用 eg:ddp_model.finish.... // 如果不是同类型也没关系，把这个当成一个普通函数，然后运行不会报错也能通过(只要含有module属性) - [By: Weijie] - 2026/03/26
    def finish_gradient_synchronization(self) -> None:
        """
        model optimizer进来, 已经完成backward了,这时候就要把梯度 all reduce, 需要optimizer干啥...
        """
        world_size = dist.get_world_size()
        for p in self.module.parameters():
            if p.grad is None:
                continue
            dist.all_reduce(p.grad, op = dist.ReduceOp.SUM)
            p.grad /= world_size

class DDPOverlapIndividualParameters(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module
        # 漏了broadcast model这一步了
        self.handles = []
        for p in self.module.parameters():
            dist.broadcast(p.data, src = 0)
            if p.requires_grad:
                p.register_post_accumulate_grad_hook(self.backward_all_reduce)

    def backward_all_reduce(self, param):
        if param.grad is None:
            return
        handle = dist.all_reduce(param.grad, op = dist.ReduceOp.SUM, async_op = True)
        self.handles.append((param, handle))
        

    def forward(self, *args, **kwargs):
        self.handles.clear()
        return self.module(*args, **kwargs)
    
    def finish_gradient_synchronization(self):
        world_size = dist.get_world_size()
        for param, handle in self.handles:
            handle.wait()
            param.grad.div_(world_size)
            
