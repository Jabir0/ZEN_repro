import torch
from extensions.cuda.functions import sparse_balance
import nvtx

class TensorPartition():
    def __init__(self, partitions, memory_scale, rehash):
        self.partitions = partitions
        self.memory_scale = memory_scale
        self.rehash = rehash
        self.device = torch.cuda.current_device()


    def balance_sparse_tensor(self, tensor):
        numel = tensor.numel()
        memory_size = numel * self.memory_scale
        memory_size = int(memory_size // self.partitions * self.partitions)
        memory = torch.zeros(memory_size, dtype=torch.int32, device=self.device)
        serial_memory_size  = int(numel // self.partitions // 10 * self.partitions)
        serial_memory = torch.zeros(serial_memory_size, dtype=torch.int32, device=self.device)
        sparse_balance(tensor, memory, serial_memory, self.partitions, rehash=self.rehash)
        return memory, serial_memory


    def get_sub_sparse_tensors(self, tensor, memory):
        with nvtx.annotate("get_sub_sparse_tensors"):
            with nvtx.annotate("chunk"):
                memory_chunks = memory.chunk(self.partitions)
            indices, values, splits = [], [], []
            for chunk in memory_chunks:
                with nvtx.annotate("sparse_indices"):
                    sparse_indices = chunk[chunk != 0]
                with nvtx.annotate("sparse_values"):
                    sparse_values = tensor[sparse_indices.type(torch.long)]
                with nvtx.annotate("append"):
                    indices.append(sparse_indices)
                    values.append(sparse_values)
                    splits.append(sparse_indices.numel())
            with nvtx.annotate("split"):
                splits = torch.tensor(splits, device=self.device)
            return indices, values, splits


    def load_balance(self, tensor):
        with nvtx.annotate("balance_sparse_tensor"):
            memory, serial_memory = self.balance_sparse_tensor(tensor)
        indices, values, splits = self.get_sub_sparse_tensors(tensor, memory)
        serial_indices, serial_values, serial_splits = self.get_sub_sparse_tensors(tensor, serial_memory)
        indices = [torch.cat([idx, serial_idx], dim=0) for idx, serial_idx in zip(indices, serial_indices)]
        values = [torch.cat([val, serial_val], dim=0) for val, serial_val in zip(values, serial_values)]
        splits = torch.tensor([split + serial_split for split, serial_split in zip(splits, serial_splits)], device=splits.device)
        return indices, values, splits