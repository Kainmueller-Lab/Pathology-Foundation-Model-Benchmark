import os
import torch.distributed as dist
import torch


def init_distributed():
    # Initializes the distributed backend which will take care of synchronizing nodes/GPUs
    global RANK, WORLD_SIZE, LOCAL_RANK, LOCAL_WORLD_SIZE
    # only works with torch.distributed.launch // torch.run
    dist.init_process_group(backend="nccl")
    RANK = dist.get_rank()
    WORLD_SIZE = dist.get_world_size()
    LOCAL_WORLD_SIZE = dist.get_world_size()
    RANK = dist.get_rank(group=None)
    LOCAL_RANK = RANK % LOCAL_WORLD_SIZE

    # this will make all .cuda() calls work properly
    torch.cuda.set_device(LOCAL_RANK)

    # synchronizes all the threads to reach this point before moving on
    dist.barrier()
    print("Distributed training environment successfully initialized")
    return LOCAL_RANK, LOCAL_WORLD_SIZE, RANK, WORLD_SIZE


def all_gather(t_local, WORLD_SIZE):
    """
    Gather tensors from all process, and return the concatenated results
    Args:
        t_local (torch.Tensor): tensor to be gathered, assumes the same size across all processes
        WORLD_SIZE (int): number of processes
    Returns:
        output_tensor (torch.Tensor): gathered tensor
    """
    output_tensor = [torch.zeros_like(t_local) for _ in range(WORLD_SIZE)]
    dist.all_gather(output_tensor, t_local)
    output_tensor = torch.cat(output_tensor, dim=0)
    return output_tensor
