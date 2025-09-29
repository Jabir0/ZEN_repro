import torch
import torch.distributed as dist
from DDPbackend import DDPBackend
from tensor_load_balance import TensorPartition
from extensions.cuda.functions import hash_index
import nvtx

class LoadBalanceAllreduce():
    def __init__(self, process_group, global_ranks, logger, memory_scale=1, rehash=3):
        self.name = "LoadBalanceAllreduce"
        self.ddp = DDPBackend(process_group, global_ranks, logger)
        self.device = torch.cuda.current_device()
        self.process_group = process_group
        self.partitions = self.process_group.size()
        self.local_rank = dist.get_rank(group=self.process_group)
        self.tensor_partition = self.get_tensor_partition_obj(memory_scale, rehash)
        self.logger = logger
        self.mapping = {}


    def get_tensor_partition_obj(self, memory_scale, rehash):
        return TensorPartition(partitions=self.partitions, memory_scale=memory_scale, rehash=rehash)


    def stack_tensors(self, tensors, shape=-1):
        if isinstance(tensors, list):
            return torch.cat(tensors).reshape(shape)
        else:
            return tensors.reshape(shape)


    def decode_sparse_tensors(self, indices_tensors, values_tensors, alltoall_dense_size):
        if isinstance(indices_tensors, list):
            dtype = values_tensors[0].dtype
            _dense_tensor = torch.zeros(alltoall_dense_size, dtype=dtype, device=self.device)
            indices = self.stack_tensors(indices_tensors).type(torch.int64)
            values = self.stack_tensors(values_tensors)
            _dense_tensor.scatter_add_(0, indices, values)
            return _dense_tensor
        else:
            dtype = values_tensors.dtype
            _dense_tensor = torch.zeros(alltoall_dense_size, dtype=dtype, device=self.device)
            _dense_tensor.scatter_add_(0, indices_tensors.type(torch.int64), values_tensors)
            return _dense_tensor


    def encode_sparse_tensor(self, tensor):
        sparse_indices = torch.nonzero(tensor, as_tuple=True)[0]
        sparse_values = tensor[sparse_indices]
        return sparse_indices, sparse_values

    def calculate_mapping(self, numel):
        indices = torch.zeros(numel, dtype=torch.int32, device=self.device)
        hash_index(indices, self.partitions)
        mapping = [(indices==i).nonzero(as_tuple=True)[0] for i in range(self.partitions)]
        return mapping


    def encode_bitmap(self, sparse_indices, numel):
        def sorted_isin(sorted_tensor, sorted_values):
            indices = torch.searchsorted(sorted_tensor, sorted_values)
            result = torch.zeros(len(sorted_tensor), dtype=torch.uint8, device=sorted_tensor.device)
            result[indices] = 1
            
            return result
        with nvtx.annotate("torch.isin"):
            mask = sorted_isin(self.mapping[numel][self.local_rank], sparse_indices).to(torch.uint8)
        with nvtx.annotate("padding"):
            # Pad the mask to make its length a multiple of 8
            num_bits = mask.numel()
            num_bytes = (num_bits + 7) // 8  # Round up to the nearest byte
            padded_length = num_bytes * 8
            if num_bits < padded_length:
                padding = torch.zeros(padded_length - num_bits, dtype=torch.uint8, device=mask.device)
                mask = torch.cat([mask, padding])
            # Reshape mask to a 2D tensor where each row represents one byte (8 bits)
            mask = mask.view(-1, 8)
        with nvtx.annotate("packing"):

            # Create a bit shift tensor [1, 2, 4, 8, 16, 32, 64, 128] for packing
            bit_shifts = 2 ** torch.arange(8, dtype=torch.uint8, device=mask.device)
            
            # Multiply and sum across rows to convert each row of 8 bits into a byte
            packed_bitmap = (mask * bit_shifts).sum(dim=1).to(torch.uint8)
        return packed_bitmap

    def decode_bitmap(self, bitmap_indices, numel):
        with nvtx.annotate("allgather mapping_tensor"):
            mapping_tensor = self.mapping[numel]
        indices = []
        with nvtx.annotate("bitmap_indices"):
            for i, bitmap_tensor in enumerate(bitmap_indices):
                bit_shifts = 2 ** torch.arange(8, device=bitmap_tensor.device, dtype=torch.uint8)
                with nvtx.annotate("unpacked_bits"):
                    unpacked_bits = ((bitmap_tensor.unsqueeze(-1) & bit_shifts) > 0).to(torch.bool)
                with nvtx.annotate("mask"):
                    mask = unpacked_bits.view(-1)[:mapping_tensor[i].numel()]
                indices.append(mapping_tensor[i][mask])
        return indices

    def zen_sparse_comm(self, tensor, method):
        numel = tensor.numel()
        if method == 1:
            # direct allgatherv
            with nvtx.annotate("encode_sparse_tensor"):
                sparse_indices, sparse_values = self.encode_sparse_tensor(tensor)
            with nvtx.annotate("allgatherv indices"):
                indices_tensors = self.ddp.allgather_padding(sparse_indices)
            with nvtx.annotate("allgatherv values"):
                values_tensors = self.ddp.allgather_padding(sparse_values)
            with nvtx.annotate("decode_sparse_tensors"):
                return self.decode_sparse_tensors(indices_tensors, values_tensors, numel)
        else:

            # load balance then alltoallv then allgatherv
            with nvtx.annotate("load_balance"):
                indices, values, splits = self.tensor_partition.load_balance(tensor)
            with nvtx.annotate("allgather"):
                gather_splits = self.ddp.allgather(splits)
            with nvtx.annotate("alltoallv indices"):
                indices_tensors = self.ddp.alltoallv(indices, gather_splits)
            with nvtx.annotate("alltoallv values"):
                values_tensors = self.ddp.alltoallv(values, gather_splits)
            with nvtx.annotate("decode_sparse_tensors"):
                _dense_tensor = self.decode_sparse_tensors(indices_tensors, values_tensors, numel)
            with nvtx.annotate("encode_sparse_tensor"):
                sparse_indices, sparse_values = self.encode_sparse_tensor(_dense_tensor)
            if method == 2:
                with nvtx.annotate("allgather indices"):
                    indices_tensors = self.ddp.allgather_padding(sparse_indices)
            else:
                if numel not in self.mapping:
                    with nvtx.annotate("calculate_mapping"):
                        self.mapping[numel] = self.calculate_mapping(numel)
                with nvtx.annotate("allgather_bitmap"):
                    with nvtx.annotate("encode_bitmap"):
                        bitmap_indices = self.encode_bitmap(sparse_indices, numel)
                    with nvtx.annotate("allgatherv"):
                        bitmap_indices_gather = self.ddp.allgather_padding(bitmap_indices)
                    with nvtx.annotate("decode_bitmap"):
                        indices_tensors = self.decode_bitmap(bitmap_indices_gather, numel)
            with nvtx.annotate("allgather indices"):
                values_tensors = self.ddp.allgather_padding(sparse_values)
            with nvtx.annotate("decode_sparse_tensors"):
                tensor = self.decode_sparse_tensors(indices_tensors, values_tensors, numel)
            return tensor

