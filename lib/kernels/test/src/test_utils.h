#ifndef _FLEXFLOW_KERNELS_TEST_UTILS
#define _FLEXFLOW_KERNELS_TEST_UTILS

#include "kernels/device.h"
#include "kernels/ff_handle.h"
#include "kernels/local_allocator.h"
#include <algorithm>
#include <memory>
#include <random>
#include <vector>

template <typename T>
void allocate_ptrs(std::vector<T **> &gpu_data_ptrs,
                   std::vector<size_t> const &num_elements,
                   Allocator &allocator) {
  for (size_t i = 0; i < gpu_data_ptrs.size(); ++i) {
    *gpu_data_ptrs[i] =
        static_cast<T *>(allocator.allocate(num_elements[i] * sizeof(float)));
  }
}

GenericTensorAccessorW getRandomFilledAccessorW(TensorShape const &shape,
                                                Allocator &allocator);

GenericTensorAccessorW getFilledAccessorW(TensorShape const &shape,
                                          Allocator &allocator,
                                          float val);

void cleanup_test(cudaStream_t &stream, PerDeviceFFHandle &handle);

TensorShape get_float_tensor_shape(FFOrdered<size_t> dims);

TensorShape get_double_tensor_shape(FFOrdered<size_t> dims);

template <typename T>
std::vector<T> fill_host_data(void *gpu_data, size_t num_elements) {
  std::vector<T> local_data(num_elements);
  checkCUDA(cudaMemcpy(local_data.data(),
                       gpu_data,
                       local_data.size() * sizeof(T),
                       cudaMemcpyDeviceToHost));
  return local_data;
}

template <typename T>
inline bool contains_non_zero(std::vector<T> &data) {
  for (auto &val : data) {
    if (val != 0) {
      return true;
    }
  }
  return false;
}

inline void setPerDeviceFFHandle(PerDeviceFFHandle *handle) {
  cudnnCreate(&handle->dnn);
  cublasCreate(&handle->blas);
  handle->workSpaceSize = 1024 * 1024;
  cudaMalloc(&handle->workSpace, handle->workSpaceSize);
  handle->allowTensorOpMathConversion = true;
}
#endif
