"""
backward pass - all reduce == single process training outcome
"""
from collections import defaultdict
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors

class DDPIndividualParameters(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module
        # NOTE: broadcast操作, src = 0 传播p.data ,src != 0 ， 将本地的的p.data 由传播的覆盖 - [By: Weijie] - 2026/03/26
        # NOTE: 如果想改变映射对象，则可以取中间变量桥接 eg: A <- tensor = tensor -> B - [By: Weijie] - 2026/03/26
        with torch.no_grad():
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
        with torch.no_grad():
            for p in self.module.parameters():
                dist.broadcast(p, src = 0)
        
        for p in self.module.parameters():
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
            
class DDPOverlapBucketed(nn.Module):
    def __init__(self, module, bucket_size_mb):
        super().__init__()
        self.module = module
        self.bucket_size_mb = bucket_size_mb

        self.handles = []
        # NOTE: broadcast & bucket - [By: Weijie] - 2026/03/31
        with torch.no_grad():
            for p in self.module.parameters():
                dist.broadcast(p, src = 0)

        for p in self.module.parameters():
            if p.requires_grad:
                p.register_post_accumulate_grad_hook(self._account)


    def set_buckets(self):
        current_buckets = []
        current_bucket_size = 0
        self.buckets = []
        self.p_bucket = {}
        self.bucket_cnt = defaultdict(int)

        bucket_size_bytes = int(self.bucket_size_mb * 1024 * 1024)

        bucket_id = 0
        for p in reversed(list(self.module.parameters())):
            if not p.requires_grad:
                continue
            # TODO: 这是在做什么，函数的用法 - [By: Weijie] - 2026/03/31
            p_bytes = p.numel() * p.element_size()

            if current_buckets and current_bucket_size + p_bytes > bucket_size_bytes:
                self.buckets.append(current_buckets)
                bucket_id += 1
                current_buckets = []
                current_bucket_size = 0
            
            current_buckets.append(p)
            # NOTE: 不使用p，而是用唯一标识id(p) - [By: Weijie] - 2026/03/31
            self.p_bucket[id(p)] = bucket_id
            current_bucket_size += p_bytes
        
        if current_buckets:
            self.buckets.append(current_buckets)

        
    def _account(self, param):
        bucket_id = self.p_bucket[id(param)]
        self.bucket_cnt[bucket_id] += 1

        # NOTE: ready时就开始all reduce - [By: Weijie] - 2026/03/31
        if self.bucket_cnt[bucket_id] == len(self.buckets[bucket_id]):
            params_bucket = self.buckets[bucket_id]
            grads_bucket = [p.grad for p in params_bucket]
            flatten_tensor = _flatten_dense_tensors(grads_bucket)
            handle = dist.all_reduce(flatten_tensor, op = dist.ReduceOp.SUM, async_op = True)
            # NOTE: 还需要同时返回flatten tensor给出地址源 - [By: Weijie] - 2026/03/31
            self.handles.append((bucket_id, handle, flatten_tensor))
            del self.bucket_cnt[bucket_id]

 
    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)
    
    def finish_gradient_synchronization(self):
        # TODO: 这里思维还不够衔接，需要注意的是这里时backward后面的计算，通常的状态会是需要通信了，要理解清楚这部分在训练流程中的位置有助于衔接 - [By: Weijie] - 2026/03/31
        world_size = dist.get_world_size()
        for bucket_id, handle, flatten_tensor in self.handles:
            params_bucket = self.buckets[bucket_id]
            grads_bucket = [p.grad for p in params_bucket]
            handle.wait()
            flatten_tensor.div_(world_size)
            unflatten_tensor = _unflatten_dense_tensors(flatten_tensor, grads_bucket)
            for idx, grad in enumerate(unflatten_tensor):
                params_bucket[idx].grad.copy_(grad)
        
        self.handles.clear()
        self.bucket_cnt.clear()

            
