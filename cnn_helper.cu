#include "cnn_helper.h"
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

__global__
void reluBackward(float *grad_ptr, const float *input, int n)
{
  CUDA_KERNEL_LOOP(i, n)
  {
    grad_ptr[i] = (input[i] > 0.0f) ? grad_ptr[i] : 0;
  }
}

__global__
void apply_add(float *data_ptr, float *replica_ptr, size_t size)
{
  CUDA_KERNEL_LOOP(i, size)
  {
    data_ptr[i] += replica_ptr[i];   
  }
}

__host__
void updateGAS(float *data, size_t replica_size, int num_replica, float learning_rate)
{
  // Step 1: gater gradients to the first replica
  for (int i = 1; i < num_replica; i++) {
    float *replica = data + i * replica_size;
    apply_add<<<GET_BLOCKS(replica_size), CUDA_NUM_THREADS>>>(
        data, replica, replica_size);
  }
  // Step 2: scale the first replica
  float scale_factor = 1.0f / num_replica * (-learning_rate);
  scale_kernel<<<GET_BLOCKS(replica_size), CUDA_NUM_THREADS>>>(
      data, replica_size, 0, scale_factor);
  // Step 3: copy parameters back to each replica
  for (int i = 1; i < num_replica; i++) {
    float *replica = data + i * replica_size;
    checkCUDA(cudaMemcpyAsync(replica, data,
                              replica_size * sizeof(float),
                              cudaMemcpyDeviceToDevice));
  }
}
