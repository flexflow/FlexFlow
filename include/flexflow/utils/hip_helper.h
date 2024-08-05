#ifndef _FLEXFLOW_HIP_HELPER_H_
#define _FLEXFLOW_HIP_HELPER_H_
#include "flexflow/accessor.h"
#include "flexflow/ffconst.h"
#include "legion.h"
#include <hipblas/hipblas.h>
#include <miopen/miopen.h>
#ifdef FF_USE_NCCL
#include <rccl/rccl.h>
#endif

#define FatalError(s)                                                          \
  do {                                                                         \
    std::stringstream _where, _message;                                        \
    _where << __FILE__ << ':' << __LINE__;                                     \
    _message << std::string(s) + "\n" << __FILE__ << ':' << __LINE__;          \
    std::cerr << _message.str() << "\nAborting...\n";                          \
    assert(false);                                                             \
    exit(1);                                                                   \
  } while (0)

#define checkCUDNN(status)                                                     \
  do {                                                                         \
    std::stringstream _error;                                                  \
    if (status != miopenStatusSuccess) {                                       \
      _error << "CUDNN failure: " << status;                                   \
      FatalError(_error.str());                                                \
    }                                                                          \
  } while (0)

#define checkCURAND(status)                                                    \
  do {                                                                         \
    std::stringstream _error;                                                  \
    if (status != HIPRAND_STATUS_SUCCESS) {                                    \
      _error << "CURAND failure: " << status;                                  \
      FatalError(_error.str());                                                \
    }                                                                          \
  } while (0)

#define checkCUDA(status)                                                      \
  do {                                                                         \
    std::stringstream _error;                                                  \
    if (status != 0) {                                                         \
      _error << "Cuda failure: " << status;                                    \
      FatalError(_error.str());                                                \
    }                                                                          \
  } while (0)

#ifdef FF_USE_NCCL
#define checkNCCL(cmd)                                                         \
  do {                                                                         \
    ncclResult_t r = cmd;                                                      \
    if (r != ncclSuccess) {                                                    \
      printf("Failed, NCCL error %s:%d '%s'\n",                                \
             __FILE__,                                                         \
             __LINE__,                                                         \
             ncclGetErrorString(r));                                           \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)
#endif

// CUDA: grid stride looping
#define CUDA_KERNEL_LOOP(i, n)                                                 \
  for (Legion::coord_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (n);     \
       i += blockDim.x * gridDim.x)

// Use 1024 threads per block, which requires cuda sm_2x or above
int const CUDA_NUM_THREADS = 1024;
int const BLOCK_SIZE_LIMIT = 32768;

// CUDA: number of blocks for threads.
inline int GET_BLOCKS(int const N) {
  int ret = (N + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS;
  return (ret > BLOCK_SIZE_LIMIT) ? BLOCK_SIZE_LIMIT : ret;
}

template <typename DT>
__global__ void scale_kernel(DT *ptr, Legion::coord_t size, DT a, DT b);

__global__ void ones_kernel(float *ptr, Legion::coord_t size);

template <typename DT>
__global__ void assign_kernel(DT *ptr, Legion::coord_t size, DT value);

template <typename DT>
__global__ void copy_kernel(DT *dst, const DT *src, Legion::coord_t size);

template <typename DT>
__global__ void copy_kernel_discrete(DT *dst,
                                     const DT *src,
                                     Legion::coord_t size,
                                     size_t *index);

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
                                   hipStream_t stream);

__host__ void sigmoid_backward_kernel(DataType data_type,
                                      void *output_grad_ptr,
                                      void const *output_ptr,
                                      size_t output_size,
                                      hipStream_t stream);

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
void print_tensor(T const *ptr,
                  size_t num_elements,
                  char const *prefix,
                  int shard_id = 0);
template <typename T>
void print_beam_tensor(T const *ptr,
                       size_t num_elements,
                       int skip,
                       int channel,
                       char const *prefix);

template <typename T>
void save_tensor(T const *ptr, size_t num_elements, char const *file_name);

template <typename T>
T *copy_tensor_dev_to_host(T const *ptr, size_t num_elements);

template <typename T>
void copy_tensor_dev_to_host(T const *ptr, T *dst, size_t num_elements);

template <typename T>
void copy_tensor_host_to_dev(T *dst, T const *src, size_t num_elements);

miopenStatus_t
    cudnnSetTensorDescriptorFromDomain(miopenTensorDescriptor_t tensor,
                                       Legion::Domain domain,
                                       DataType data_type = DT_FLOAT);

miopenStatus_t
    cudnnSetTensorDescriptorFromDomain4SoftMax(miopenTensorDescriptor_t tensor,
                                               Legion::Domain domain,
                                               DataType data_type = DT_FLOAT);

hipblasDatatype_t ff_to_cuda_datatype(DataType type);

miopenDataType_t ff_to_cudnn_datatype(DataType type);
#ifdef FF_USE_NCCL
ncclDataType_t ff_to_nccl_datatype(DataType type);
#endif

void handle_unimplemented_hip_kernel(OperatorType op_type);
#endif
void check_device_vs_host_ptr(void const *maybe_devicePtr);
void check_ptr_alignment(void const *ptr);
