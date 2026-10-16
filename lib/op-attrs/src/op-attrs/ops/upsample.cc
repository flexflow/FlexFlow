#include "op-attrs/ops/upsample.h"
#include "op-attrs/parallel_tensor_dims.h"
#include "op-attrs/parallel_tensor_shape.h"
#include "op-attrs/tensor_dims.h"
#include <libassert/assert.hpp>

namespace FlexFlow {

static void check_mode(UpsampleAttrs const &attrs) {
  ASSERT(attrs.mode == UpsampleMode::NEAREST,
         "Currently Upsample is only supports mode {}. "
         "If you need other support for other modes, please create an issue.",
         UpsampleMode::NEAREST);
}

TensorShape upsample_get_output_shape(UpsampleAttrs const &attrs,
                                      TensorShape const &input_shape) {
  check_mode(attrs);

  ASSERT(get_num_dims(input_shape.dims) == num_tensor_dims_t{4_n},
         "Currently Upsample only supports 4-dimensional input tensors (i.e., "
         "NCHW tensors). "
         "If you need other support for other tensor shapes, please create an "
         "issue.");

  TensorShape result = input_shape;
  dim_at_idx(result.dims, relative_ff_dim_t{-1}) *= attrs.scale_factor;
  dim_at_idx(result.dims, relative_ff_dim_t{-2}) *= attrs.scale_factor;
  return result;
}

ParallelTensorDimDegrees upsample_get_output_parallel_dim_degrees(
    UpsampleAttrs const &attrs,
    ParallelTensorDimDegrees const &input_dim_degrees) {
  check_mode(attrs);

  return input_dim_degrees;
}

ParallelTensorShape
    upsample_get_output_parallel_shape(UpsampleAttrs const &attrs,
                                       ParallelTensorShape const &input_shape) {
  TensorShape unpar =
      upsample_get_output_shape(attrs, get_reduced_shape(input_shape));
  ParallelTensorDimDegrees degrees = upsample_get_output_parallel_dim_degrees(
      attrs, get_parallel_degrees(input_shape));

  return lift_shape_to_parallel_with_degrees(unpar, degrees);
}

} // namespace FlexFlow
