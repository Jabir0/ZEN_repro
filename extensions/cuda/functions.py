import torch
import functions_cuda as func

def sparse_balance(tensor, memory, serial_memory, partitions, rehash):
    assert tensor.is_cuda and tensor.dtype == torch.bfloat16
    assert memory.is_cuda and memory.dtype == torch.int32
    assert serial_memory.is_cuda and serial_memory.dtype == torch.int32
    numel = serial_memory.numel()
    numel_per_part = numel // partitions
    offsets = [i*numel_per_part for i in range(partitions)]
    offsets_tensor = torch.tensor(offsets, dtype=torch.int32, device=tensor.device)
    func.sparse_balance(tensor, memory, serial_memory, offsets_tensor, partitions, rehash)

def hash_index(index, partitions):
    func.hash_index(index, partitions)