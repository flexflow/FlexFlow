#include "cuda_helper.h"
__global__
void scale_kernel(float* ptr, coord_t size, float a, float b)
{
  CUDA_KERNEL_LOOP(i, size)
  {
    ptr[i] = (b - a) * ptr[i] + a;
  }
}

__global__
void ones_kernel(float* ptr, coord_t size)
{
  CUDA_KERNEL_LOOP(i, size)
  {
    ptr[i] = 1.0f;
  }
}

template<typename DT>
__global__
void assign_kernel(DT* ptr, coord_t size, DT value)
{
  CUDA_KERNEL_LOOP(i, size)
  {
    ptr[i] = value;
  }
}

template<typename DT>
__global__
void copy_kernel(DT* dst, const DT* src, coord_t size)
{
  CUDA_KERNEL_LOOP(i, size)
  {
    dst[i] = src[i];
  }
}

__global__
void reluBackward(float *grad_ptr, const float *output, int n)
{
  CUDA_KERNEL_LOOP(i, n)
  {
    grad_ptr[i] = (output[i] > 0.0f) ? grad_ptr[i] : 0;
  }
}

__global__
void apply_add(float *data_ptr, const float *replica_ptr, size_t size)
{
  CUDA_KERNEL_LOOP(i, size)
  {
    data_ptr[i] += replica_ptr[i];   
  }
}

__global__
void apply_add_with_scale(float *data_ptr, const float *grad_ptr,
                          size_t size, float scale)
{
  CUDA_KERNEL_LOOP(i, size)
  {
    data_ptr[i] += grad_ptr[i] * scale;
  }
}

__host__
void updateGAS(float* para_ptr, const float* grad_ptr, size_t replica_size,
               int num_replica, float learning_rate)
{
  // Step 1: gater gradients to the first replica
  for (int i = 1; i < num_replica; i++) {
    const float *replica = grad_ptr + i * replica_size;
    apply_add<<<GET_BLOCKS(replica_size), CUDA_NUM_THREADS>>>(
        (float*)grad_ptr, replica, replica_size);
  }
  // Step 2: scale the first replica
  float scale_factor = 1.0f / num_replica * (-learning_rate);
  apply_add_with_scale<<<GET_BLOCKS(replica_size), CUDA_NUM_THREADS>>>(
      para_ptr, grad_ptr, replica_size, scale_factor);
}


template __global__ void assign_kernel<float>(float* ptr, coord_t size, float value);
template __global__ void assign_kernel<int32_t>(int32_t* ptr, coord_t size, int32_t value);
template __global__ void assign_kernel<int64_t>(int64_t* ptr, coord_t size, int64_t value);

template __global__ void copy_kernel<float>(float* dst, const float* src, coord_t size);
template __global__ void copy_kernel<int>(int* dst, const int* src, coord_t size);
