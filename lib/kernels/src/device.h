#ifndef _FLEXFLOW_KERNELS_SRC_DEVICE_H
#define _FLEXFLOW_KERNELS_SRC_DEVICE_H

#include "kernels/array_shape.h"
#include "kernels/device.h"
#include "op-attrs/datatype.h"
#include "op-attrs/operator_type.h"
#include <cstddef>

namespace FlexFlow {

#if defined(FF_USE_CUDA)
#include <cuda_fp16.h>
#elif defined(FF_USE_HIP_CUDA)
#include <cuda_fp16.h>
#elif defined(FF_USE_HIP_ROCM)
#include <hip/hip_fp16.h>
#endif

#if defined(FF_USE_CUDA) || defined(FF_USE_HIP_CUDA)
#define FF_CUDNN_STATUS_SUCCESS CUDNN_STATUS_SUCCESS
#define FF_CURAND_STATUS_SUCCESS CURAND_STATUS_SUCCESS
#define FF_CUBLAS_STATUS_SUCCESS CUBLAS_STATUS_SUCCESS
#elif defined(FF_USE_HIP_ROCM)
#define FF_CUDNN_STATUS_SUCCESS miopenStatusSuccess
#define FF_CURAND_STATUS_SUCESS HIPRAND_STATUS_SUCCESS
#else
#error "Unknown device"
#endif

#define checkCUDNN(status)                                                     \
  do {                                                                         \
    std::stringstream _error;                                                  \
    if (status != FF_CUDNN_STATUS_SUCCESS) {                                   \
      _error << "CUDNN failure: " << status;                                   \
      FatalError(_error.str());                                                \
    }                                                                          \
  } while (0)

#define checkCURAND(status)                                                    \
  do {                                                                         \
    std::stringstream _error;                                                  \
    if (status != FF_CURAND_STATUS_SUCCESS) {                                  \
      _error << "CURAND failure: " << status;                                  \
      FatalError(_error.str());                                                \
    }                                                                          \
  } while (0)

#define checkCUBLAS(status)                                                    \
  do {                                                                         \
    std::stringstream _error;                                                  \
    if (status != FF_CUBLAS_STATUS_SUCCESS) {                                  \
      _error << "CUBLAS failure: " << status;                                  \
      FatalError(_error.str());                                                \
    }                                                                          \
  } while (0)

// CUDA: grid stride looping
#define CUDA_KERNEL_LOOP(i, n)                                                 \
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (n);              \
       i += blockDim.x * gridDim.x)

// Use 1024 threads per block, which requires cuda sm_2x or above
int const CUDA_NUM_THREADS = 1024;
int const BLOCK_SIZE_LIMIT = 32768;

// CUDA: number of blocks for threads.
inline int GET_BLOCKS(int const N) {
  int ret = (N + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS;
  return (ret > BLOCK_SIZE_LIMIT) ? BLOCK_SIZE_LIMIT : ret;
}

__global__ void scale_kernel(float *ptr, size_t size, float a, float b);

__global__ void ones_kernel(float *ptr, size_t size);

template <typename DT>
__global__ void assign_kernel(DT *ptr, size_t size, DT value);

template <typename DT>
__global__ void copy_kernel(DT *dst, const DT *src, size_t size);

template <typename T>
__global__ void add_kernel(T *data_ptr, T const *grad_ptr, size_t size);

template <typename T>
__global__ void reluBackward(T *grad_ptr, T const *input, size_t n);

template <typename T>
__global__ void sigmoid_backward_kernel(T *grad_ptr, T const *input, size_t n);

__host__ void relu_backward_kernel(DataType data_type,
                                   void *output_grad_ptr,
                                   void const *output_ptr,
                                   size_t output_size,
                                   cudaStream_t stream);

__host__ void sigmoid_backward_kernel(DataType data_type,
                                      void *output_grad_ptr,
                                      void const *output_ptr,
                                      size_t output_size,
                                      cudaStream_t stream);

template <typename DT>
__global__ void apply_add_with_scale(DT *data_ptr,
                                     const DT *grad_ptr,
                                     size_t size,
                                     DT scale);

__global__ void
    gelu_forward_kernel(size_t size, float B, float C, float *input);

// Use by concat and split
__global__ void add_with_stride(float *output,
                                float const *input,
                                int num_blocks,
                                int output_blk_size,
                                int input_blk_size);
__global__ void copy_with_stride(float *output,
                                 float const *input,
                                 int num_blocks,
                                 int output_blk_size,
                                 int input_blk_size);

__host__ void updateGAS(float *para_ptr,
                        float const *grad_ptr,
                        size_t replica_size,
                        int num_replica,
                        float learning_rate);

template <typename T>
void print_tensor(T const *ptr, size_t num_elements, char const *prefix);

ffStatus_t cudnnSetTensorDescriptorFromArrayShape(ffTensorDescriptor_t tensor,
                                                  ArrayShape const &shape);

ffDataType_t ff_to_cuda_datatype(DataType type);

ffCudnnDataType_t ff_to_cudnn_datatype(DataType type);

void handle_unimplemented_kernel(OperatorType op_type);

} // namespace FlexFlow

#endif
