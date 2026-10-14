#include "kernels/accessors_are_within_epsilon.h"
#include "kernels/map_tensor_accessors.h"
#include "kernels/tensor_accessor_binary_ops.h"
#include "kernels/tensor_accessor_reductions.h"

namespace FlexFlow {

bool accessors_are_within_epsilon(GenericTensorAccessorR const &accessor_a,
                                  GenericTensorAccessorR const &accessor_b,
                                  std::optional<float> epsilon) {
  TensorShape shape = require_same(accessor_a.shape, accessor_b.shape);

  ASSERT(shape.data_type == DataType::FLOAT);

  float resolved_epsilon =
      epsilon.value_or(std::numeric_limits<float>::epsilon());

  Allocator cpu_allocator = create_local_cpu_memory_allocator();
  GenericTensorAccessorW diff = tensor_accessor_elementwise_subtract(
      accessor_a, accessor_b, cpu_allocator);
  GenericTensorAccessorW within_epsilon = map_tensor_accessor(
      diff,
      [&](float x) { return std::abs(x) < resolved_epsilon; },
      cpu_allocator);
  return tensor_accessor_all(within_epsilon);
}

} // namespace FlexFlow
