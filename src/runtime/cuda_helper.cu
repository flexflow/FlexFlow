#include "cuda_helper.h"
#include "model.h"

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

template<unsigned DIM, typename T>
__host__
void print_tensor(const T* ptr, Rect<DIM> rect, const char* prefix)
{
  // device synchronize to make sure the data are ready
  checkCUDA(cudaDeviceSynchronize());
  T* host_ptr;
  checkCUDA(cudaHostAlloc(&host_ptr, sizeof(T) * rect.volume(),
                          cudaHostAllocPortable | cudaHostAllocMapped));
  checkCUDA(cudaMemcpy(host_ptr, ptr, sizeof(T) * rect.volume(),
                       cudaMemcpyDeviceToHost));
  checkCUDA(cudaDeviceSynchronize());
  int idx = 0;
  printf("%s", prefix);
  for (PointInRectIterator<DIM> it(rect); it(); it++, idx++) {
    printf(" %.4lf", (float)host_ptr[idx]);
  }
  printf("\n");
  checkCUDA(cudaFreeHost(host_ptr));
}

cudnnStatus_t cudnnSetTensorDescriptorFromDomain(cudnnTensorDescriptor_t tensor, Domain domain)
{
  int dims[MAX_DIM];
  switch (domain.get_dim()) {
    case 1:
    {
      Rect<1> rect = domain;
      dims[0] = rect.hi[0] - rect.lo[0] + 1;
      return cudnnSetTensor4dDescriptor(tensor, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, dims[0], 1, 1, 1);
    }
    case 2:
    {
      Rect<2> rect = domain;
      dims[0] = rect.hi[0] - rect.lo[0] + 1;
      dims[1] = rect.hi[1] - rect.lo[1] + 1;
      return cudnnSetTensor4dDescriptor(tensor, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, dims[1], dims[0], 1, 1);
    }
    case 3:
    {
      Rect<3> rect = domain;
      dims[0] = rect.hi[0] - rect.lo[0] + 1;
      dims[1] = rect.hi[1] - rect.lo[1] + 1;
      dims[2] = rect.hi[2] - rect.lo[2] + 1;
      return cudnnSetTensor4dDescriptor(tensor, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, dims[2], dims[1], dims[0], 1);
    }
    case 4:
    {
      Rect<4> rect = domain;
      dims[0] = rect.hi[0] - rect.lo[0] + 1;
      dims[1] = rect.hi[1] - rect.lo[1] + 1;
      dims[2] = rect.hi[2] - rect.lo[2] + 1;
      dims[3] = rect.hi[3] - rect.lo[3] + 1;
      return cudnnSetTensor4dDescriptor(tensor, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, dims[3], dims[2], dims[1], dims[0]);
    }
    default:
      assert(false && "Unsupported dim number");
  }
  return CUDNN_STATUS_BAD_PARAM;
}

template __global__ void assign_kernel<float>(float* ptr, coord_t size, float value);
template __global__ void assign_kernel<int32_t>(int32_t* ptr, coord_t size, int32_t value);
template __global__ void assign_kernel<int64_t>(int64_t* ptr, coord_t size, int64_t value);

template __global__ void copy_kernel<float>(float* dst, const float* src, coord_t size);
template __global__ void copy_kernel<int>(int* dst, const int* src, coord_t size);

template __host__ void print_tensor<1, float>(const float* ptr, Rect<1> rect, const char* prefix);
template __host__ void print_tensor<2, float>(const float* ptr, Rect<2> rect, const char* prefix);
template __host__ void print_tensor<3, float>(const float* ptr, Rect<3> rect, const char* prefix);
template __host__ void print_tensor<4, float>(const float* ptr, Rect<4> rect, const char* prefix);
template __host__ void print_tensor<2, long>(const long* ptr, Rect<2> rect, const char* prefix);
