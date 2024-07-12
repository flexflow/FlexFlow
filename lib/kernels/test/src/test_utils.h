#ifndef _FLEXFLOW_KERNELS_TEST_UTILS
#define _FLEXFLOW_KERNELS_TEST_UTILS

#include "kernels/device.h"
#include "kernels/local_cpu_allocator.h"
#include "kernels/local_cuda_allocator.h"
#include "kernels/managed_ff_stream.h"
#include "kernels/managed_per_device_ff_handle.h"
#include <random>

enum class GpuDirection {
  HostToDevice = 0,
  DeviceToHost = 1,
  DeviceToDevice = 2
};

template <typename DT>
void transfer_memory(DT *dst,
                     const DT *src,
                     size_t num_elements,
                     GpuDirection gpu_dir,
                     bool cpu_memory) {
  size_t bytes = num_elements * sizeof(DT);

  if (cpu_memory) {
    memcpy(dst, src, bytes);
  } else {
    switch (gpu_dir) {
      case GpuDirection::HostToDevice:
        checkCUDA(cudaMemcpy(dst, src, bytes, cudaMemcpyHostToDevice));
        break;
      case GpuDirection::DeviceToHost:
        checkCUDA(cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToHost));
        break;
      case GpuDirection::DeviceToDevice:
        checkCUDA(cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToDevice));
        break;
    }
  }
}

GenericTensorAccessorW create_random_filled_accessor_w(TensorShape const &shape,
                                                       Allocator &allocator,
                                                       bool on_host = false);

template <typename DT>
GenericTensorAccessorW create_filled_accessor_w(TensorShape const &shape,
                                                Allocator &allocator,
                                                DT val,
                                                bool on_host = false) {
  GenericTensorAccessorW accessor = allocator.allocate_tensor(shape);
  size_t volume = accessor.shape.num_elements();
  std::vector<DT> host_data(volume, val);

  transfer_memory(static_cast<DT *>(accessor.ptr),
                  host_data.data(),
                  volume,
                  GpuDirection::HostToDevice,
                  on_host);

  return accessor;
}

template <typename IDT, typename ODT, typename F>
GenericTensorAccessorW create_transformed_accessor_w(TensorShape const &shape,
                                                     Allocator &allocator,
                                                     F transform,
                                                     bool on_host = false) {
  GenericTensorAccessorW accessor = allocator.allocate_tensor(shape);
  size_t volume = accessor.shape.get_volume();
  std::vector<IDT> input_data(volume);
  std::vector<ODT> output_data(volume);

  std::transform(
      input_data.begin(), input_data.end(), output_data.begin(), transform);

  transfer_memory(static_cast<ODT *>(accessor.ptr),
                  output_data.data(),
                  volume,
                  GpuDirection::HostToDevice,
                  on_host);

  return accessor;
}

template <DataType DT>
GenericTensorAccessorW
    copy_tensor_between_memories(GenericTensorAccessorR accessor,
                                 TensorShape const &shape,
                                 Allocator &allocator,
                                 bool src_on_host = false) {
  GenericTensorAccessorW copied_accessor = allocator.allocate_tensor(shape);

  size_t volume = accessor.shape.get_volume();
  GpuDirection gpu_dir =
      src_on_host ? GpuDirection::HostToDevice : GpuDirection::DeviceToHost;

  transfer_memory(
      copied_accessor.get<DT>(), accessor.get<DT>(), volume, gpu_dir, false);

  return copied_accessor;
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

template <DataType DT>
std::vector<real_type<DT>> load_accessor_data(GenericTensorAccessorR accessor,
                                              bool on_device = true) {
  int volume = accessor.shape.get_volume();

  using T = real_type<DT>;
  std::vector<T> local_data(volume);
  T const *src_ptr = accessor.get<DT>();

  if (on_device) {
    checkCUDA(cudaMemcpy(local_data.data(),
                         src_ptr,
                         volume * sizeof(T),
                         cudaMemcpyDeviceToHost));
  } else {
    std::memcpy(local_data.data(), src_ptr, volume * sizeof(T));
  }

  return local_data;
}

template <typename T>
bool contains_non_zero(std::vector<T> &data) {
  return !all_of(data, [](T const &val) { return val == 0; });
}

#endif
