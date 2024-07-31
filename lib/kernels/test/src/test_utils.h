#ifndef _FLEXFLOW_KERNELS_TEST_UTILS
#define _FLEXFLOW_KERNELS_TEST_UTILS

#include "kernels/device.h"
#include "kernels/local_cpu_allocator.h"
#include "kernels/local_cuda_allocator.h"
#include "kernels/managed_ff_stream.h"
#include "kernels/managed_per_device_ff_handle.h"
#include <random>

template <typename DT>
void transfer_memory(GenericTensorAccessorW dst_accessor,
                     const DT *src,
                     AllocLocation src_loc) {
  size_t bytes = dst_accessor.shape.get_volume() * sizeof(DT);
  AllocLocation dst_loc =
      dst_accessor.on_device ? AllocLocation::DEVICE : AllocLocation::HOST;

  if (src_loc == AllocLocation::HOST && dst_loc == AllocLocation::HOST) {
    memcpy(dst_accessor.ptr, src, bytes);
  } else if (src_loc == AllocLocation::HOST &&
             dst_loc == AllocLocation::DEVICE) {
    checkCUDA(cudaMemcpy(dst_accessor.ptr, src, bytes, cudaMemcpyHostToDevice));
  } else if (src_loc == AllocLocation::DEVICE &&
             dst_loc == AllocLocation::HOST) {
    checkCUDA(cudaMemcpy(dst_accessor.ptr, src, bytes, cudaMemcpyDeviceToHost));
  } else {
    checkCUDA(
        cudaMemcpy(dst_accessor.ptr, src, bytes, cudaMemcpyDeviceToDevice));
  }
}

template <DataType DT>
GenericTensorAccessorW create_random_filled_accessor_w(TensorShape const &shape,
                                                       Allocator &allocator) {
  assert(shape.data_type == DataType::FLOAT ||
         shape.data_type == DataType::DOUBLE);
  using T = real_type<DT>;

  GenericTensorAccessorW accessor = allocator.allocate_tensor(shape);
  accessor.on_device =
      (allocator.alloc_location == AllocLocation::DEVICE) ? true : false;

  std::vector<T> host_data(accessor.shape.num_elements());
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<T> dist(-1.0, 1.0);

  for (auto &val : host_data) {
    val = dist(gen);
  }

  transfer_memory(accessor, host_data.data(), AllocLocation::HOST);

  return accessor;
}

template <DataType DT>
GenericTensorAccessorR create_random_filled_accessor_r(TensorShape const &shape,
                                                       Allocator &allocator) {
  GenericTensorAccessorW accessor =
      create_random_filled_accessor_w<DT>(shape, allocator);

  return read_only_accessor_from_write_accessor(accessor);
}

template <typename DT>
GenericTensorAccessorW create_filled_accessor_w(TensorShape const &shape,
                                                Allocator &allocator,
                                                DT val) {
  GenericTensorAccessorW accessor = allocator.allocate_tensor(shape);
  accessor.on_device =
      (allocator.alloc_location == AllocLocation::DEVICE) ? true : false;

  size_t volume = accessor.shape.get_volume();
  std::vector<DT> host_data(volume, val);

  transfer_memory(accessor, host_data.data(), AllocLocation::HOST);

  return accessor;
}

template <typename IDT, typename ODT, typename F>
GenericTensorAccessorW create_transformed_accessor_w(TensorShape const &shape,
                                                     Allocator &allocator,
                                                     F transform) {
  GenericTensorAccessorW accessor = allocator.allocate_tensor(shape);
  accessor.on_device =
      (allocator.alloc_location == AllocLocation::DEVICE) ? true : false;

  size_t volume = accessor.shape.get_volume();
  std::vector<IDT> input_data(volume);
  std::vector<ODT> output_data(volume);

  std::transform(
      input_data.begin(), input_data.end(), output_data.begin(), transform);

  transfer_memory(accessor, output_data.data(), AllocLocation::HOST);

  return accessor;
}

template <DataType DT>
GenericTensorAccessorW
    copy_tensor_between_memories(GenericTensorAccessorR accessor,
                                 Allocator &allocator) {
  TensorShape shape = get_tensor_shape(accessor.shape, accessor.data_type);
  GenericTensorAccessorW copied_accessor = allocator.allocate_tensor(shape);
  copied_accessor.on_device =
      (allocator.alloc_location == AllocLocation::DEVICE) ? true : false;

  AllocLocation src_loc =
      accessor.on_device ? AllocLocation::DEVICE : AllocLocation::HOST;

  transfer_memory(copied_accessor, accessor.get<DT>(), src_loc);

  return copied_accessor;
}

TensorShape make_tensor_shape_from_legion_dims(FFOrdered<size_t> dims,
                                               DataType DT);

template <DataType DT>
std::vector<real_type<DT>> load_accessor_data(GenericTensorAccessorR accessor) {
  using T = real_type<DT>;

  int volume = accessor.shape.get_volume();
  std::vector<T> local_data(volume);
  T const *src_ptr = accessor.get<DT>();

  if (accessor.on_device) {
    checkCUDA(cudaMemcpy(local_data.data(),
                         src_ptr,
                         volume * sizeof(T),
                         cudaMemcpyDeviceToHost));
  } else {
    memcpy(local_data.data(), src_ptr, volume * sizeof(T));
  }

  return local_data;
}

template <DataType DT>
std::vector<real_type<DT>> load_accessor_data(GenericTensorAccessorW accessor) {
  using T = real_type<DT>;

  int volume = accessor.shape.get_volume();
  std::vector<T> local_data(volume);
  T const *src_ptr = accessor.get<DT>();

  if (accessor.on_device) {
    checkCUDA(cudaMemcpy(local_data.data(),
                         src_ptr,
                         volume * sizeof(T),
                         cudaMemcpyDeviceToHost));
  } else {
    memcpy(local_data.data(), src_ptr, volume * sizeof(T));
  }

  return local_data;
}

template <typename T>
bool contains_non_zero(std::vector<T> &data) {
  return !all_of(data, [](T const &val) { return val == 0; });
}

#endif
