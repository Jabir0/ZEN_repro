#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <cuda_bf16.h>

namespace {

#define FORCE_INLINE
__device__ static inline FORCE_INLINE uint64_t rotl32 ( uint32_t x, int8_t r) {
  return (x << r) | (x >> (32 - r));
}


#define	ROTL32(x,y)	rotl32(x,y)

//-----------------------------------------------------------------------------
// Block read - if your platform needs to do endian-swapping or can only
// handle aligned reads, do the conversion here

#define getblock(p, i) (p[i])
#define SEED 43

//-----------------------------------------------------------------------------
// Finalization mix - force all bits of a hash block to avalanche
__device__ static inline FORCE_INLINE uint32_t fmix32 ( uint32_t h )
{
  h ^= h >> 16;
  h *= 0x85ebca6b;
  h ^= h >> 13;
  h *= 0xc2b2ae35;
  h ^= h >> 16;

  return h;
}

// murmurhash for integer only
// https://github.com/aappleby/smhasher/blob/master/src/MurmurHash3.cpp
__device__ uint32_t MurmurHash3_x86_32 (
        const int key, 
        const uint32_t seed) {
    uint32_t h1 = seed;
    uint32_t c1 = 0xcc9e2d51;
    uint32_t c2 = 0x1b873593;
    
    uint32_t k1 = key;
    
    k1 *= c1;
    k1 = ROTL32(k1, 15);
    k1 *= c2;
    
    h1 ^= k1;
    h1 = ROTL32(h1, 13); 
    h1 = h1*5+0xe6546b64;

    //----------
    // tail

    const uint8_t * tail = (const uint8_t*)key;
    k1 = 0;

    int len = 4;
    switch(len & 3) {
        case 3: k1 ^= tail[2] << 16;
        case 2: k1 ^= tail[1] << 8;
        case 1: k1 ^= tail[0];
        k1 *= c1; k1 = ROTL32(k1, 15); k1 *= c2; h1 ^= k1;
    };

    //----------
    // finalization
    h1 ^= len;
    h1 = fmix32(h1);
    return h1;
} 


__device__ uint32_t universal_sparse_hash(const int key, const uint32_t seed) {
    uint32_t d = 573729859;
    uint32_t a = 0xcc9e2d51;
    uint32_t b = 0x1b873593;
    uint32_t c = seed;

    return d*key*key*key + a*key*key + b*key + c;
}


__device__ uint32_t sparse_hash(const int key, const uint32_t seed) {
    // return MurmurHash3_x86_32(key, seed);
    return universal_sparse_hash(key, seed);
}

// seeds = [10001861, 100703, 5002639, 607037]

__global__ void sparse_balance_cuda_kernel (__nv_bfloat16* input, int input_size, int* memory, int memory_size, int* serial_memory, int* offset, int partitions) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < input_size) {
        if (__bfloat162float(input[tid]) != 0.0f) {
            int index = sparse_hash(tid, 10001861) % memory_size;
            if (memory[index] == 0) {
                memory[index] = tid;
            } else {
                const int numel_per_part = memory_size / partitions;
                const int part_id = index / numel_per_part;                
                int i = atomicAdd(&(offset[part_id]), 1);
                serial_memory[i] = tid;
            }
        }
    }
}


__global__ void sparse_balance_two_rehash_cuda_kernel (__nv_bfloat16* input, int input_size, int* memory, int memory_size, int* serial_memory, int* offset, int partitions) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < input_size) {
        if (__bfloat162float(input[tid]) != 0.0f) {
            int index = sparse_hash(tid, 10001861) % memory_size;
            if (memory[index]) {
                const int numel_per_part = memory_size / partitions;
                const int part_id = index / numel_per_part;
                const int start_pos = part_id * numel_per_part;
                index = sparse_hash(tid, 100703) % numel_per_part;  
                if (memory[index + start_pos] == 0) {
                    memory[index + start_pos] = tid;
                } else {
                    int i = atomicAdd(&(offset[part_id]), 1);
                    serial_memory[i] = tid;
                }
            } else {
                memory[index] = tid;
            }
        }
    }
}



__global__ void sparse_balance_three_rehash_cuda_kernel (__nv_bfloat16* input, int input_size, int* memory, int memory_size, int* serial_memory, int* offset, int partitions) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < input_size) {
        if (__bfloat162float(input[tid]) != 0.0f) {
            int index = sparse_hash(tid, 10001861) % memory_size;
            if (memory[index]) {
                const int numel_per_part = memory_size / partitions;
                const int part_id = index / numel_per_part;
                const int start_pos = part_id * numel_per_part;
                index = sparse_hash(tid, 100703) % numel_per_part;
                if (memory[index + start_pos]) {
                    index = sparse_hash(tid, 5002639) % numel_per_part;
                    if (memory[index + start_pos] == 0) {
                        memory[index + start_pos] = tid;
                    } else {
                        int i = atomicAdd(&(offset[part_id]), 1);
                        serial_memory[i] = tid;
                    }
                }
                else {
                    memory[index + start_pos] = tid;
                }
            } else {
                memory[index] = tid;
            }
        }
    }
}

__global__ void hash_index_cuda_kernel (int* input, int input_size, int partitions) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < input_size) {
        int index = sparse_hash(tid, 10001861) % input_size;
        const int numel_per_part = input_size / partitions;
        const int part_id = index / numel_per_part;
        input[tid] = part_id;
    }
}
} // namespace

void sparse_balance_cuda (
        void* input, 
        int input_size,
        void* memory, 
        int memory_size,
        void* serial_memory,
        void* offset,
        int partitions,
        int rehash) {
    const int threads = 512;
    const int blocks = (input_size + threads - 1) / threads;

    if (rehash == 2) {
        sparse_balance_two_rehash_cuda_kernel<<<blocks, threads>>>((__nv_bfloat16*)input, input_size, (int*)memory, memory_size, (int*)serial_memory, (int*)offset, partitions);
    } else if (rehash == 3) {
        sparse_balance_three_rehash_cuda_kernel<<<blocks, threads>>>((__nv_bfloat16*)input, input_size, (int*)memory, memory_size, (int*)serial_memory, (int*)offset, partitions);
    } else {
        sparse_balance_cuda_kernel<<<blocks, threads>>>((__nv_bfloat16*)input, input_size, (int*)memory, memory_size, (int*)serial_memory, (int*)offset, partitions);
    }
}

void hash_index_cuda(void* input, int input_size, int partitions) {
    const int threads = 512;
    const int blocks = (input_size + threads - 1) / threads;
    hash_index_cuda_kernel<<<blocks, threads>>>((int*)input, input_size, partitions);
}