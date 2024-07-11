#ifndef _FLEXFLOW_KERNELS_TEST_UTILS
#define _FLEXFLOW_KERNELS_TEST_UTILS

#include "kernels/device.h"
#include "kernels/local_cpu_allocator.h"
#include "kernels/local_cuda_allocator.h"
#include "kernels/managed_ff_stream.h"
#include "kernels/managed_per_device_ff_handle.h"
#include <random>

template <typename DT>
GenericTensorAccessorW create_random_filled_accessor_w(TensorShape const &shape,
                                                       Allocator &allocator,
                                                       bool cpu_fill = false) {
  GenericTensorAccessorW accessor = allocator.allocate_tensor(shape);
  size_t volume = accessor.shape.num_elements();
  std::vector<DT> host_data(volume);
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<DT> dist(-1.0f, 1.0f);

  for (auto &val : host_data) {
    val = dist(gen);
  }

  if (cpu_fill) {
    memcpy(accessor.ptr, host_data.data(), host_data.size() * sizeof(DT));
  } else {
    checkCUDA(cudaMemcpy(accessor.ptr,
                         host_data.data(),
                         host_data.size() * sizeof(DT),
                         cudaMemcpyHostToDevice));
  }

  return accessor;
}

template <typename DT>
GenericTensorAccessorW create_filled_accessor_w(TensorShape const &shape,
                                                Allocator &allocator,
                                                DT val,
                                                bool cpu_fill = false) {
  GenericTensorAccessorW accessor = allocator.allocate_tensor(shape);
  size_t volume = accessor.shape.num_elements();
  std::vector<DT> host_data(volume, val);

  if (cpu_fill) {
    memcpy(accessor.ptr, host_data.data(), host_data.size() * sizeof(DT));
  } else {
    checkCUDA(cudaMemcpy(accessor.ptr,
                         host_data.data(),
                         host_data.size() * sizeof(DT),
                         cudaMemcpyHostToDevice));
  }

  return accessor;
}

template <typename DT>
GenericTensorAccessorW create_iota_filled_accessor_w(TensorShape const &shape,
                                                     Allocator &allocator,
                                                     bool cpu_fill = false) {
  GenericTensorAccessorW accessor = allocator.allocate_tensor(shape);
  size_t volume = accessor.shape.num_elements();
  std::vector<DT> host_data(volume);

  for (size_t i = 0; i < volume; i++) {
    host_data[i] = i;
  }

  if (cpu_fill) {
    memcpy(accessor.ptr, host_data.data(), host_data.size() * sizeof(DT));
  } else {
    checkCUDA(cudaMemcpy(accessor.ptr,
                         host_data.data(),
                         host_data.size() * sizeof(DT),
                         cudaMemcpyHostToDevice));
  }

  return accessor;
}

template <DataType DT>
TensorShape make_tensor_shape_from_legion_dims(FFOrdered<size_t> dims) {
  return TensorShape{
      TensorDims{
          dims,
      },
      DT,
  };
}

template <typename DT>
std::vector<DT> load_data_to_host_from_device(GenericTensorAccessorR accessor) {
  int volume = accessor.shape.get_volume();

  std::vector<DT> local_data(volume);
  checkCUDA(cudaMemcpy(local_data.data(),
                       accessor.ptr,
                       local_data.size() * sizeof(DT),
                       cudaMemcpyDeviceToHost));
  return local_data;
}

template <typename T>
bool contains_non_zero(std::vector<T> &data) {
  return !all_of(data, [](T const &val) { return val == 0; });
}

#endif
