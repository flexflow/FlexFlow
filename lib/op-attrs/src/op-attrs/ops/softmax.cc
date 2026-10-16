#include "op-attrs/ops/softmax.h"
#include "op-attrs/parallel_tensor_shape.h"
#include "op-attrs/tensor_dims.h"
#include "op-attrs/tensor_shape.h"

namespace FlexFlow {

TensorShape get_output_shape(SoftmaxAttrs const &attrs,
                             TensorShape const &input_shape) {
  ASSERT(attrs.dim.value < get_num_dims(input_shape.dims),
         fmt::format("get_output_shape for Softmax received out-of-bounds "
                     "attrs.dim {} for input tensor shape {}",
                     attrs.dim,
                     input_shape));

  return input_shape;
}

ParallelTensorDimDegrees get_output_parallel_dim_degrees(
    SoftmaxAttrs const &attrs, ParallelTensorDimDegrees const &input_degrees) {
  ASSERT(input_degrees.sum_degree.value == 1,
         fmt::format("Expected sum degree 1, but received sum degree {}",
                     input_degrees.sum_degree));
  ASSERT(
      input_degrees.discard_copy_degree.value == 1,
      fmt::format(
          "Expected discard copy degree 1, but received discard copy degree {}",
          input_degrees.discard_copy_degree));
  ASSERT(
      input_degrees.shard_degrees.at(attrs.dim) == 1,
      fmt::format("Expected parallel degree of Softmax dimension {} to be 1, "
                  "but received input degrees {}",
                  attrs.dim,
                  input_degrees));

  return input_degrees;
}

ParallelTensorShape get_output_shape(SoftmaxAttrs const &attrs,
                                     ParallelTensorShape const &input_shape) {
  TensorShape unpar = get_output_shape(attrs, get_reduced_shape(input_shape));

  ParallelTensorDimDegrees output_degrees =
      get_output_parallel_dim_degrees(attrs, get_parallel_degrees(input_shape));

  return lift_shape_to_parallel_with_degrees(unpar, output_degrees);
}

} // namespace FlexFlow
