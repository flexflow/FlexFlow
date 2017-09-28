#ifndef _LEGION_CNN_HELPER_H_
#define _LEGION_CNN_HELPER_H_
#include "legion.h"
// CUDA: grid stride looping
#define CUDA_KERNEL_LOOP(i, n) \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); i += blockDim.x * gridDim.x)

// Use 1024 threads per block, which requires cuda sm_2x or above
const int CUDA_NUM_THREADS = 1024;

// CUDA: number of blocks for threads.
inline int GET_BLOCKS(const int N)
{
  return (N + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS;
}
using namespace Legion;

__global__
void scale_kernel(float* ptr, coord_t size, float a, float b);

__global__
void ones_kernel(float* ptr, coord_t size);

__global__
void reluBackward(float* grad_ptr, const float* input, int n);

#endif
