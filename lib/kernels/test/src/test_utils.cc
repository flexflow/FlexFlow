#include "test_utils.h"

GenericTensorAccessorW create_random_filled_accessor_w(TensorShape const &shape,
                                                       Allocator &allocator,
                                                       bool on_host) {
  GenericTensorAccessorW accessor = allocator.allocate_tensor(shape);
  size_t volume = accessor.shape.num_elements();
  std::vector<float> host_data(volume);
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

  for (auto &val : host_data) {
    val = dist(gen);
  }

  transfer_memory(static_cast<float *>(accessor.ptr),
                  host_data.data(),
                  volume,
                  GpuDirection::HostToDevice,
                  on_host);

  return accessor;
}
