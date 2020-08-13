#ifndef _LEGION_CNN_HELPER_H_
#define _LEGION_CNN_HELPER_H_
#include "legion.h"
#include <cudnn.h>

#define FatalError(s) do {                                             \
    std::stringstream _where, _message;                                \
    _where << __FILE__ << ':' << __LINE__;                             \
    _message << std::string(s) + "\n" << __FILE__ << ':' << __LINE__;  \
    std::cerr << _message.str() << "\nAborting...\n";                  \
    exit(1);                                                           \
} while(0)

#define checkCUDNN(status) do {                                        \
    std::stringstream _error;                                          \
    if (status != CUDNN_STATUS_SUCCESS) {                              \
      _error << "CUDNN failure: " << cudnnGetErrorString(status);      \
      FatalError(_error.str());                                        \
    }                                                                  \
} while(0)

#define checkCURAND(status) do {                                       \
    std::stringstream _error;                                          \
    if (status != CURAND_STATUS_SUCCESS) {                             \
      _error << "CURAND failure: " << status;                          \
      FatalError(_error.str());                                        \
    }                                                                  \
} while(0)

#define checkCUDA(status) do {                                         \
    std::stringstream _error;                                          \
    if (status != 0) {                                                 \
      _error << "Cuda failure: " << status;                            \
      FatalError(_error.str());                                        \
    }                                                                  \
} while(0)

// CUDA: grid stride looping
#define CUDA_KERNEL_LOOP(i, n) \
  for (coord_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); i += blockDim.x * gridDim.x)

// Use 1024 threads per block, which requires cuda sm_2x or above
const int CUDA_NUM_THREADS = 1024;
const int BLOCK_SIZE_LIMIT = 32768;

// CUDA: number of blocks for threads.
inline int GET_BLOCKS(const int N)
{
  int ret = (N + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS;
  return (ret > BLOCK_SIZE_LIMIT) ? BLOCK_SIZE_LIMIT : ret;
}
using namespace Legion;

__global__
void scale_kernel(float* ptr, coord_t size, float a, float b);

__global__
void ones_kernel(float* ptr, coord_t size);

template<typename DT>
__global__
void assign_kernel(DT* ptr, coord_t size, DT value);

template<typename DT>
__global__
void copy_kernel(DT* dst, const DT* src, coord_t size);

__global__
void reluBackward(float* grad_ptr, const float* input, int n);

__global__
void apply_add_with_scale(float *data_ptr, const float *grad_ptr,
                          size_t size, float scale);

__host__
void updateGAS(float* para_ptr, const float* grad_ptr, size_t replica_size,
               int num_replica, float learning_rate);

template<unsigned DIM, typename T>
void print_tensor(const T* ptr, Rect<DIM> rect, const char* prefix);

cudnnStatus_t cudnnSetTensorDescriptorFromDomain(cudnnTensorDescriptor_t tensor,
                                                 Legion::Domain domain);
#endif
