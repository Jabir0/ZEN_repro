import torch
import torch.distributed as dist

class DDPBackend():
    def __init__(self, process_group, global_ranks, logger):
        self.process_group = process_group
        self.group_size = dist.get_world_size(group=self.process_group)
        self.local_rank = dist.get_rank(group=self.process_group)
        self.global_ranks = global_ranks
        self.logger = logger


    # when tensors on all GPUs have the same size
    def allgather(self, tensor, async_op=True):
        if self.group_size == 1:
            return tensor
        ret = [torch.empty_like(tensor) for _ in range(self.group_size)]
        if async_op:
            handle = dist.all_gather(ret, tensor, group=self.process_group, async_op=True)
            handle.wait()
        else:
            dist.all_gather(ret, tensor, group=self.process_group)
        return ret
    
    # allgather when tensors on all GPUs have the different size
    def allgather_padding(self, tensor, async_op=True):
        numel = torch.tensor(tensor.numel(), dtype=torch.int32, device=tensor.device)
        tensor_sizes = self.allgather(numel, async_op)
        max_size = max(tensor_sizes)
        if tensor.numel() < max_size:
            padding = torch.zeros(max_size - tensor.numel(), dtype=tensor.dtype, device=tensor.device)
            tensor = torch.cat([tensor, padding])

        output_padding = [torch.empty(max_size, dtype=tensor.dtype, device=tensor.device) for _ in range(self.group_size)]
        if async_op:
            handle = dist.all_gather(output_padding, tensor, group=self.process_group, async_op=True)
            handle.wait()
        else:
            dist.all_gather(output_padding, tensor, group=self.process_group)
        output = [o[:s] for o, s in zip(output_padding, tensor_sizes)]
        return output


    # when tensors on all GPUs have the different size,
    # this function may have deadlock issue when group_size is large
    def allgatherv(self, tensor): 
        if self.group_size == 1:
            return tensor

        numel = torch.tensor(tensor.numel(), dtype=torch.int32, device=tensor.device)
        tensor_sizes = self.allgather(numel)
        ret = torch.empty(sum(tensor_sizes), dtype=tensor.dtype, device=tensor.device)
        ret = list(torch.split(ret, tensor_sizes))

        req = []
        for rank in range(self.group_size):
            if rank != self.local_rank:
                if rank > self.local_rank:
                    req.append(dist.isend(tensor=tensor, group=self.process_group, dst=self.global_ranks[rank]))
                    req.append(dist.irecv(tensor=ret[rank], group=self.process_group, src=self.global_ranks[rank]))
                else:
                    req.append(dist.irecv(tensor=ret[rank], group=self.process_group, src=self.global_ranks[rank]))
                    req.append(dist.isend(tensor=tensor, group=self.process_group, dst=self.global_ranks[rank]))
            else:
                ret[rank] = tensor
        
        for r in req:
            r.wait()
        return ret

    def alltoallv(self, tensors, gather_splits):
        if self.group_size == 1:
            return tensors
        assert(len(tensors) == self.group_size)

        tensor_sizes = [s[self.local_rank] for s in gather_splits]
        ret = torch.empty(sum(tensor_sizes), dtype=tensors[0].dtype, device=tensors[0].device)
        ret = list(torch.split(ret, tensor_sizes))

        dist.all_to_all(ret, tensors, group=self.process_group, async_op=True).wait()
        return ret
