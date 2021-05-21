#include "cuda_helper.h"
#include "model.h"


#ifdef LEGION_USE_HIP
#ifdef __HIP_PLATFORM_NVCC__
extern "C" {
cudaStream_t hipGetTaskStream();
}

cudaError_t get_legion_stream(cudaStream_t *stream)
{
#ifdef DISABLE_LEGION_CUDA_HIJACK
  *stream = (cudaStream_t)0;
#else
  *stream = hipGetTaskStream();
#endif
  return cudaSuccess;
}
#endif
#else
cudaError_t get_legion_stream(cudaStream_t *stream)
{
#ifdef DISABLE_LEGION_CUDA_HIJACK
  *stream = (cudaStream_t)0;
  return cudaSuccess;
#else
  return cudaStreamCreate(stream);
#endif
}
#endif

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
void gelu_forward_kernel(size_t size, const float B, const float C, float* input)
{
  CUDA_KERNEL_LOOP(i, size)
  {
    const float in = input[i];
    const float cdf = 0.5f + 0.5f * tanh(in * (C * in * in + B));
    input[i] = in * cdf;
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

__global__
void add_with_stride(float* output,
                     const float* input,
                     int num_blocks,
                     int output_blk_size,
                     int input_blk_size)
{
  int min_blk_size = min(output_blk_size, input_blk_size);
  CUDA_KERNEL_LOOP(i, num_blocks * min_blk_size)
  {
    int blk_idx = i / min_blk_size;
    int blk_offset = i % min_blk_size;
    int input_offset = blk_idx * input_blk_size + blk_offset;
    int output_offset = blk_idx * output_blk_size + blk_offset;
    output[output_offset] += input[input_offset];
  }
}

__global__
void copy_with_stride(float* output,
                      const float* input,
                      int num_blocks,
                      int output_blk_size,
                      int input_blk_size)
{
  int min_blk_size = min(output_blk_size, input_blk_size);
  CUDA_KERNEL_LOOP(i, num_blocks * min_blk_size)
  {
    int blk_idx = i / min_blk_size;
    int blk_offset = i % min_blk_size;
    int input_offset = blk_idx * input_blk_size + blk_offset;
    int output_offset = blk_idx * output_blk_size + blk_offset;
    output[output_offset] = input[input_offset];
  }
}



__host__
void updateGAS(float* para_ptr, const float* grad_ptr, size_t replica_size,
               int num_replica, float learning_rate)
{
  cudaStream_t stream;
  checkCUDA(get_legion_stream(&stream));
  // Step 1: gater gradients to the first replica
  for (int i = 1; i < num_replica; i++) {
    const float *replica = grad_ptr + i * replica_size;
    apply_add<<<GET_BLOCKS(replica_size), CUDA_NUM_THREADS, 0, stream>>>(
        (float*)grad_ptr, replica, replica_size);
  }
  // Step 2: scale the first replica
  float scale_factor = 1.0f / num_replica * (-learning_rate);
  apply_add_with_scale<<<GET_BLOCKS(replica_size), CUDA_NUM_THREADS, 0, stream>>>(
      para_ptr, grad_ptr, replica_size, scale_factor);
}

#ifdef DEADCODE
template<unsigned DIM, typename T>
__host__
void print_tensor(const T* ptr, Rect<DIM> rect, const char* prefix)
{
  // device synchronize to make sure the data are ready
  // checkCUDA(cudaDeviceSynchronize());
  T* host_ptr;
  checkCUDA(cudaHostAlloc(&host_ptr, sizeof(T) * rect.volume(),
                          cudaHostAllocPortable | cudaHostAllocMapped));
  checkCUDA(cudaMemcpy(host_ptr, ptr, sizeof(T) * rect.volume(),
                       cudaMemcpyDeviceToHost));
  // checkCUDA(cudaDeviceSynchronize());
  int idx = 0;
  printf("%s", prefix);
  for (PointInRectIterator<DIM> it(rect); it(); it++, idx++) {
    printf(" %.4lf", (float)host_ptr[idx]);
    if (idx >= 16) break;
  }
  printf("\n");
  checkCUDA(cudaFreeHost(host_ptr));
}
#endif

template<typename T>
__host__
void print_tensor(const T* ptr, size_t num_elements, const char* prefix)
{
  // device synchronize to make sure the data are ready
  // checkCUDA(cudaDeviceSynchronize());
  T* host_ptr;
  checkCUDA(cudaHostAlloc(&host_ptr, sizeof(T) * num_elements,
                          cudaHostAllocPortable | cudaHostAllocMapped));
  checkCUDA(cudaMemcpy(host_ptr, ptr, sizeof(T) * num_elements,
                       cudaMemcpyDeviceToHost));
  // checkCUDA(cudaDeviceSynchronize());
  int idx = 0;
  printf("%s", prefix);
  for (idx=0; idx < num_elements; idx++) {
    printf(" %.4lf", (float)host_ptr[idx]);
    if (idx >= 16) break;
  }
  printf("\n");
  checkCUDA(cudaFreeHost(host_ptr));
}

cudnnStatus_t cudnnSetTensorDescriptorFromDomain(cudnnTensorDescriptor_t tensor, Domain domain)
{
  int dims[MAX_TENSOR_DIM];
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

template __host__ void print_tensor<float>(const float* ptr, size_t rect, const char* prefix);
template __host__ void print_tensor<long>(const long* ptr, size_t rect, const char* prefix);
