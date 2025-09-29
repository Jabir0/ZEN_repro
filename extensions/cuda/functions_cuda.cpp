#include <torch/extension.h>
#include <vector>
#include <iostream>

// CUDA forward declarations
void sparse_balance_cuda (void* input, int input_size, void* memory, int memory_size, void* serial_memory, void* offset, int partitions, int rehash);
void hash_index_cuda(void* input, int input_size, int partitions);
// C++ interface

// NOTE: AT_ASSERT has become AT_CHECK on master after 0.4.
#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

void sparse_balance (
        torch::Tensor input,
        torch::Tensor memory,
        torch::Tensor serial_memory,
        torch::Tensor offset,
        int partitions, 
        int rehash) {
    CHECK_INPUT(input);
    CHECK_INPUT(memory);

    int input_size = input.numel();
    int memory_size = memory.numel();
    sparse_balance_cuda(input.data_ptr(), input_size, memory.data_ptr(), memory_size, serial_memory.data_ptr(), offset.data_ptr(), partitions, rehash);
}

void hash_index (
        torch::Tensor input,
        int partitions) {
    CHECK_INPUT(input);
    int input_size = input.numel();
    hash_index_cuda(input.data_ptr(), input_size, partitions);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("sparse_balance", &sparse_balance, "Balance sparse tensors (CUDA)");
    m.def("hash_index", &hash_index, "Hash index (CUDA)");
}