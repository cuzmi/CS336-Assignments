import os
import time
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

def setup(rank, world_size, device):
    # device -> host & port
    os.environ['MASTER_ADDR'] = "localhost" 
    os.environ['MASTER_PORT'] = '29500'
    if device == 'cpu':
        dist.init_process_group(
            backend = 'gloo', 
            rank = rank, 
            world_size = world_size
        )
        device = torch.device('cpu')
    elif device == 'gpu':
        # NOTE: setup 2. 进程分配到GPU上，通过rank绑定 - [By: Weijie] - 2026/03/25
        torch.cuda.set_device(rank)
        # NOTE: setup 3. 进程进行分组 - [By: Weijie] - 2026/03/25
        dist.init_process_group(
            backend = 'nccl', 
            rank = rank, 
            world_size = world_size
        )
        device = torch.device(f'cuda:{rank}')
    
    return device

def build_data(size, device):
    
    numel = size * 1024 * 1024 // 4
    return torch.ones((numel,), dtype = torch.float32, device = device)

def benchmarking_allreduce(tensor, device_type, warmup = 5, iters = 20):
    for _ in range(warmup):
        dist.all_reduce(tensor, op = dist.ReduceOp.SUM)
        if device_type == 'gpu': 
            # NOTE: 细节 1：设置进程同步对应的gpu - [By: Weijie] - 2026/03/25
            torch.cuda.synchronize(tensor.device) 

    dist.barrier()

    times = []
    for _ in range(iters):
        if device_type == 'gpu':
            torch.cuda.synchronize(tensor.device)
        
        start = time.perf_counter()
        dist.all_reduce(tensor, op = dist.ReduceOp.SUM, async_op = False)

        if device_type == 'gpu':
            torch.cuda.synchronize(tensor.device)
        
        end = time.perf_counter()
        times.append(end - start)
    
    dist.barrier()

    return sum(times) / len(times)



def distributed_demo(rank, world_size, data_size, device_type):
    # NOTE: core 1： 进程控制（分配 + 分组） - [By: Weijie] - 2026/03/25
    device = setup(rank, world_size, device_type)
    # NOTE: core 2： 进程独立执行操作 - [By: Weijie] - 2026/03/25
    for size in data_size:
        data = build_data(size, device)

        avg_time = benchmarking_allreduce(data, device_type)

        if rank == 0:
            print(
                f"backend={device_type}, world_size={world_size}, "
                f"size={size}MB, avg_time={avg_time:.6f}s"
            )
    
    dist.destroy_process_group()
    


if __name__ == "__main__":
    data_size = [1,10,100,1024]
    device_type = 'cpu'
    world_size = 6
    
    # NOTE: setup 1. 创建进程，为每个进程分配rank - [By: Weijie] - 2026/03/25
    mp.spawn(fn = distributed_demo, args = (world_size, data_size, device_type), nprocs = world_size, join = True)

