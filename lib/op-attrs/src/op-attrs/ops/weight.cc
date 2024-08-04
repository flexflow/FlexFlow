#include "op-attrs/ops/weight.h"
#include "op-attrs/parallel_tensor_shape.h"

namespace FlexFlow {

TensorShape get_output_shape(WeightAttrs const &attrs) {
  return attrs.tensor_shape;
}

ParallelTensorShape get_output_parallel_tensor_shape(WeightAttrs const &attrs) {
  return lift_to_parallel(attrs.tensor_shape);
}


} // namespace FlexFlow
