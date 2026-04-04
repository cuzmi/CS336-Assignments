"""
Profile memory change brought by optimizer sharding.
"""
import time
from typing import Any, Iterable, Type

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as f
from torch.optim import AdamW, Optimizer

from cs336_basics import model
from .common import get_train_batch, setup

vocab_size = 10000
d_model = 1600
d_ff = 6400
num_layers = 48
num_heads = 25
context_length = 256
rope_theta = 10000.0
batch_size = 2


class ShardedOptimizerC(Optimizer):
    def __init__(self, params: Iterable[torch.nn.Parameter], optimizer_cls: Type[Optimizer], **kwargs: Any):
        self.rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
        self.world_size = dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1
        self.optimizer_cls = optimizer_cls
        self.optimizer_kwargs = kwargs

        self.inner_optimizer = None
        self.all_params = []
        self.param_to_owner = {}
        self.next_param_index = 0

        param_groups = self._materialize_param_groups(params)

        super().__init__(param_groups, kwargs)
        self.inner_optimizer = optimizer_cls(self.param_groups, **kwargs)
        self.inner_optimizer.param_groups.clear()
        for param_group in self.param_groups:
            self._add_sharded_group_to_inner_optimizer(param_group)

        self.state = self.inner_optimizer.state

    @staticmethod
    def _materialize_param_groups(params):
        params = list(params)
        if len(params) == 0:
            raise ValueError("optimizer got an empty parameter list")

        if isinstance(params[0], dict):
            return [{**group, "params": list(group["params"])} for group in params]
        return [{"params": params}]

    def _add_sharded_group_to_inner_optimizer(self, param_group):
        shard_group = {k: v for k, v in param_group.items() if k != "params"}
        shard_group["params"] = [
            param for param in param_group["params"] if self.param_to_owner[id(param)] == self.rank
        ]
        if shard_group["params"]:
            self.inner_optimizer.add_param_group(shard_group)

    def add_param_group(self, param_group: dict[str, Any]):
        param_group = {**param_group, "params": list(param_group["params"])}

        for param in param_group["params"]:
            if id(param) not in self.param_to_owner:
                self.param_to_owner[id(param)] = self.next_param_index % self.world_size
                self.next_param_index += 1
                self.all_params.append(param)

        super().add_param_group(param_group)

        if self.inner_optimizer is not None:
            self._add_sharded_group_to_inner_optimizer(param_group)

    def step(self, closure=None, **kwargs):
        if closure is None:
            loss = self.inner_optimizer.step(**kwargs)
        else:
            loss = self.inner_optimizer.step(closure=closure, **kwargs)

        if self.world_size == 1:
            return loss
        for param in self.all_params:
            dist.broadcast(param.data, src=self.param_to_owner[id(param)])

        return loss

    def zero_grad(self, set_to_none: bool = True):
        super().zero_grad(set_to_none=set_to_none)


class StatsCollection:
    def __init__(self, device):
        self.device = device
        self.stats = {}
        self.mutli_run = {}
        self.current_idx = 0
        self.after_building_model = None

    def memory_stats(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
            return {
                "allocated_mb": torch.cuda.memory_allocated(self.device) / 1024**2,
                "peak_allocated_mb": torch.cuda.max_memory_allocated(self.device) / 1024**2,
            }
        return {
            "allocated_mb": 0.0,
            "peak_allocated_mb": 0.0,
        }

    def stage_capture(self, stage):
        self.stats[stage] = self.memory_stats()
        if stage == "after building model":
            self.after_building_model = self.stats[stage].copy()
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)

    def finish_capture(self):
        self.mutli_run[self.current_idx] = self.stats.copy()
        self.current_idx += 1
        self.stats = {}

    def reset_runs(self):
        self.stats = {}
        self.mutli_run = {}
        self.current_idx = 0
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)

    def return_max(self):
        local_summary = {}
        if self.after_building_model is not None:
            local_summary["after building model"] = self.after_building_model.copy()

        for epoch_stats in self.mutli_run.values():
            for stage_name, stage_stats in epoch_stats.items():
                if stage_name not in local_summary:
                    local_summary[stage_name] = stage_stats.copy()
                else:
                    local_summary[stage_name]["allocated_mb"] = max(
                        local_summary[stage_name]["allocated_mb"],
                        stage_stats["allocated_mb"],
                    )
                    local_summary[stage_name]["peak_allocated_mb"] = max(
                        local_summary[stage_name]["peak_allocated_mb"],
                        stage_stats["peak_allocated_mb"],
                    )
        return local_summary


def train_one_step(LM, x, y, optimizer, device, world_size, collections) -> None:
    optimizer.zero_grad(set_to_none=True)

    if device.type == "cuda":
        torch.cuda.synchronize(device)

    logits = LM(x)
    loss = f.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        y.reshape(-1),
    )
    loss.backward()

    if device.type == "cuda":
        torch.cuda.synchronize(device)

    for p in LM.parameters():
        if p.grad is None:
            continue
        dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
        p.grad.div_(world_size)

    if device.type == "cuda":
        torch.cuda.synchronize(device)

    collections.stage_capture(stage="before optimizer step")
    optimizer.step()
    collections.stage_capture(stage="after optimizer step")


def train_main(rank, world_size, vocab_size, batch_size, context_length, warmup, mode):
    device = setup(rank, world_size)

    x, y = get_train_batch(rank, world_size, batch_size, vocab_size, context_length, device)

    LM = model.BasicsTransformerLM(
        vocab_size,
        context_length,
        d_model,
        num_layers,
        num_heads,
        d_ff,
        rope_theta,
    ).to(device)

    with torch.no_grad():
        for p in LM.parameters():
            dist.broadcast(p, src=0)

    optimizer = AdamW(params=LM.parameters(), lr=1e-4, betas=(0.99, 0.999))
    if mode == "sharded":
        optimizer = ShardedOptimizerC(params=LM.parameters(), optimizer_cls=AdamW, lr=1e-4, betas=(0.99, 0.999))

    collections = StatsCollection(device)
    collections.stage_capture(stage="after building model")
    LM.train()

    for _ in range(warmup):
        train_one_step(LM, x, y, optimizer, device, world_size, collections)

    collections.reset_runs()

    for _ in range(5):
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        start = time.perf_counter()

        train_one_step(LM, x, y, optimizer, device, world_size, collections)

        if device.type == "cuda":
            torch.cuda.synchronize(device)
        end = time.perf_counter()

        train_time = end - start
        _ = train_time
        collections.finish_capture()

    local_summary = collections.return_max()

    gathered = [None for _ in range(world_size)] if rank == 0 else None
    dist.gather_object(local_summary, object_gather_list=gathered, dst=0)

    if rank == 0:
        print(f"\n=== mode: {mode} ===")
        for stage_name in local_summary.keys():
            print(f"[{stage_name}]")
            max_allocated_mb = 0.0
            max_peak_allocated_mb = 0.0
            for r in range(world_size):
                stats = gathered[r][stage_name]
                max_allocated_mb = max(max_allocated_mb, stats["allocated_mb"])
                max_peak_allocated_mb = max(max_peak_allocated_mb, stats["peak_allocated_mb"])
                print(
                    f"rank {r}: "
                    f"allocated_mb={stats['allocated_mb']:.2f}, "
                    f"peak_allocated_mb={stats['peak_allocated_mb']:.2f}"
                )
            print(
                f"max_across_ranks: "
                f"allocated_mb={max_allocated_mb:.2f}, "
                f"peak_allocated_mb={max_peak_allocated_mb:.2f}"
            )

    dist.destroy_process_group()


if __name__ == "__main__":
    world_size = 2
    warmup = 5
    modes = ["ddp", "sharded"]
    for mode in modes:
        cfg = (world_size, vocab_size, batch_size, context_length, warmup, mode)
        mp.spawn(fn=train_main, args=cfg, nprocs=world_size, join=True)
