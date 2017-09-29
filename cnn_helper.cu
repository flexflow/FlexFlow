#include "cnn_helper.h"
__global__
void scale_kernel(float* ptr, coord_t size, float a, float b)
{
  const coord_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < size) {
    ptr[tid] = (b - a) * ptr[tid] + a;
  }
}

__global__
void ones_kernel(float* ptr, coord_t size)
{
  const coord_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < size) {
    ptr[tid] = 1.0f;
  }
}

__global__
void reluBackward(float *grad_ptr, const float *input, int n)
{
  CUDA_KERNEL_LOOP(i, n)
  {
    grad_ptr[i] = (input[i] > 0.0f) ? grad_ptr[i] : 0;
  }
}
