#ifndef _FLEXFLOW_KERNELS_TEST_UTILS
#define _FLEXFLOW_KERNELS_TEST_UTILS

#include "kernels/device.h"
#include "kernels/ff_handle.h"
#include "kernels/local_allocator.h"
#include <algorithm>
#include <memory>
#include <random>
#include <vector>

GenericTensorAccessorW create_random_filled_accessor_w(TensorShape const &shape,
                                                       Allocator &allocator);

GenericTensorAccessorW create_filled_accessor_w(TensorShape const &shape,
                                                Allocator &allocator,
                                                float val);

void fill_tensor_accessor_w(GenericTensorAccessorW accessor, float val);

void cleanup_test(cudaStream_t &stream, PerDeviceFFHandle &handle);

TensorShape make_float_tensor_shape_from_legion_dims(FFOrdered<size_t> dims);

TensorShape make_double_tensor_shape_from_legion_dims(FFOrdered<size_t> dims);

void setPerDeviceFFHandle(PerDeviceFFHandle *handle);

PerDeviceFFHandle get_per_device_ff_handle();
PerDeviceFFHandle get_mock_per_device_ff_handle();

ffStream_t create_ff_stream();

template <typename T>
std::vector<T> load_data_to_host_from_device(GenericTensorAccessorR accessor) {
  LegionTensorDims dims = accessor.shape.dims;

  int volume = product(dims);

  std::vector<T> local_data(volume);
  checkCUDA(cudaMemcpy(local_data.data(),
                       accessor.ptr,
                       local_data.size() * sizeof(T),
                       cudaMemcpyDeviceToHost));
  return local_data;
}

template <typename T>
std::vector<T> load_vector_to_host_from_device(T *gpu_ptr,
                                               size_t num_elements) {
  std::vector<T> local_data(num_elements);
  checkCUDA(cudaMemcpy(local_data.data(),
                       gpu_ptr,
                       num_elements * sizeof(T),
                       cudaMemcpyDeviceToHost));
  return local_data;
}

template <typename T>
bool contains_non_zero(std::vector<T> &data) {
  return !all_of(data, [](T const &val) { return val == 0; });
}

#endif
