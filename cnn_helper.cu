#include "cnn_helper.h"
__global__
void scale_kernel(float* ptr, coord_t size, float a, float b)
{
  const coord_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < size) {
    ptr[tid] = (b - a) * ptr[tid] + a;
  }
}

