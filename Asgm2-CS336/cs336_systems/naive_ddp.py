"""
backward pass - all reduce == single process training outcome
"""
from collections import defaultdict
from typing import Type, Any, List, Iterable
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.optim import Optimizer
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


class ShardedOptimizer(Optimizer):
    def __init__(self, params: Iterable[torch.nn.Parameter], optimizer_cls: Type[Optimizer], **kwargs: Any):
        # rank / world_size 用来决定“哪个参数归哪个进程负责更新”
        self.rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
        self.world_size = dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1
        self.optimizer_cls = optimizer_cls
        self.optimizer_kwargs = kwargs

        # inner_optimizer: 真正执行参数更新的底层 optimizer
        # all_params: 所有参数的完整顺序表，后续 broadcast 也按这个顺序来
        # param_to_owner: 记录每个参数由哪个 rank 负责更新
        # next_param_index: 给新加入的参数分配 owner 时用的全局顺序编号
        self.inner_optimizer = None
        self.all_params = []
        self.param_to_owner = {}
        self.next_param_index = 0

        # 先把输入整理成 PyTorch Optimizer 接受的 param_group 格式
        param_groups = self._materialize_param_groups(params)

        # 必须调用父类构造函数。父类内部也会调用 add_param_group，
        # 所以我们自己的 add_param_group 要能处理“构造过程中的调用”。
        super().__init__(param_groups, kwargs)

        # 这里先用完整参数组构造一个底层 optimizer，
        # 然后立刻清空它的 param_groups，再只塞回“当前 rank 负责的参数”。
        # 这样 optimizer 的 state（例如 AdamW 的 m/v）就只会为本 rank 的 shard 建立。
        self.inner_optimizer = optimizer_cls(self.param_groups, **kwargs)
        self.inner_optimizer.param_groups.clear()
        for param_group in self.param_groups:
            self._add_sharded_group_to_inner_optimizer(param_group)

        # 让外部看到的 state 就是底层 sharded optimizer 的 state
        self.state = self.inner_optimizer.state

    @staticmethod
    def _materialize_param_groups(params):
        # 兼容两种输入：
        # 1. model.parameters()
        # 2. [{"params": ..., "lr": ...}, ...]
        params = list(params)
        if len(params) == 0:
            raise ValueError("optimizer got an empty parameter list")
        if isinstance(params[0], dict):
            return [{**group, "params": list(group["params"])} for group in params]
        return [{"params": params}]

    def _add_sharded_group_to_inner_optimizer(self, param_group):
        # 保留原 param_group 中的超参数配置，只替换其中的 params
        shard_group = {k: v for k, v in param_group.items() if k != "params"}

        # 只有 owner == 当前 rank 的参数，才交给本地 optimizer 真正更新
        shard_group["params"] = [
            param for param in param_group["params"] if self.param_to_owner[id(param)] == self.rank
        ]
        if shard_group["params"]:
            self.inner_optimizer.add_param_group(shard_group)

    def add_param_group(self, param_group: dict[str, Any]):
        # 复制一份，避免后续修改调用方传入的原对象
        param_group = {**param_group, "params": list(param_group["params"])}

        # 给每个“第一次见到的参数”分配 owner。
        # 这里采用最简单的 round-robin:
        # 第 0 个参数给 rank 0，第 1 个参数给 rank 1，...
        # 这样不同 rank 负责的参数数量大致均衡。
        for param in param_group["params"]:
            if id(param) not in self.param_to_owner:
                self.param_to_owner[id(param)] = self.next_param_index % self.world_size
                self.next_param_index += 1
                self.all_params.append(param)

        # super().add_param_group 会把这组参数注册到“外层 Optimizer 视角”里；
        # 这样 zero_grad 等通用逻辑仍然能遍历到完整参数。
        super().add_param_group(param_group)

        # 注意：父类构造函数执行期间也会调用 add_param_group。
        # 那时候 inner_optimizer 还没创建好，所以这里只能在其存在后
        # 再把当前 rank 对应的 shard 注册到底层 optimizer。
        if self.inner_optimizer is not None:
            self._add_sharded_group_to_inner_optimizer(param_group)

    def step(self, closure=None, **kwargs):
        # 第一步：只更新本 rank 负责的那部分参数
        if closure is None:
            loss = self.inner_optimizer.step(**kwargs)
        else:
            loss = self.inner_optimizer.step(closure=closure, **kwargs)

        if self.world_size == 1:
            return loss

        # 第二步：把更新后的参数同步给所有 rank。
        # 每个参数都从它的 owner rank 广播出去，这样最终所有进程上的模型参数都会一致。
        for param in self.all_params:
            dist.broadcast(param.data, src=self.param_to_owner[id(param)])

        return loss

    def zero_grad(self, set_to_none: bool = True):
        # 这里直接调用父类逻辑即可，因为外层 Optimizer 持有的是“完整参数列表”。
        # 每个 rank 都会对完整模型做 forward/backward，所以所有本地参数的 grad 都需要被清掉。
        super().zero_grad(set_to_none=set_to_none)


class MinimalShardedOptimizer(Optimizer):
    """
    这是一个“只够通过当前 tests/test_sharded_optimizer.py 的最小版本”。

    它只支持最简单的调用方式：
    - 输入是 model.parameters()
    - 不额外支持 add_param_group 的完整语义
    - 核心思想仍然一样：每个 rank 只更新一部分参数，然后 broadcast 回所有 rank
    """

    def __init__(self, params: Iterable[torch.nn.Parameter], optimizer_cls: Type[Optimizer], **kwargs: Any):
        self.rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
        self.world_size = dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1

        params = list(params)
        super().__init__(params, kwargs)

        # 保存一份完整参数顺序，后续 broadcast 也按这个顺序来
        self.all_params = self.param_groups[0]["params"]

        # 先构造一个完整 optimizer，再把它缩成“只更新当前 rank 负责的参数”
        self.inner_optimizer = optimizer_cls(self.all_params, **kwargs)
        self.inner_optimizer.param_groups[0]["params"] = [
            param for i, param in enumerate(self.all_params) if i % self.world_size == self.rank
        ]
        self.state = self.inner_optimizer.state

    def step(self, closure=None):
        loss = self.inner_optimizer.step(closure)

        if self.world_size == 1:
            return loss

        # 第 i 个参数固定由 rank (i % world_size) 负责更新；
        # 更新后从 owner rank 广播给所有进程。
        for i, param in enumerate(self.all_params):
            dist.broadcast(param.data, src=i % self.world_size)

        return loss

    def zero_grad(self, set_to_none: bool = True):
        super().zero_grad(set_to_none=set_to_none)
