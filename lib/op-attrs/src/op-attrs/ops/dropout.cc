#include "op-attrs/ops/dropout.h"
#include "op-attrs/parallel_tensor_shape.h"

namespace FlexFlow {

TensorShape get_output_shape(DropoutAttrs const &,
                             TensorShape const &input_shape) {
  return input_shape;
}

ParallelTensorShape get_output_shape(DropoutAttrs const &attrs,
                                     ParallelTensorShape const &input_shape) {
  ASSERT(get_sum_degree(input_shape) == 1,
         fmt::format("Expected sum degree 1, but receieved sum degree {}",
                     get_sum_degree(input_shape)));

  ASSERT(
      get_discard_copy_degree(input_shape) == 1,
      fmt::format(
          "Expected discard copy degree 1, but received discard copy degree {}",
          get_discard_copy_degree(input_shape)));

  return input_shape;
}

} // namespace FlexFlow
