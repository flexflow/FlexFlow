#include "op-attrs/ops/reshape.h"
#include "op-attrs/parallel_tensor_dim_degrees.h"
#include "op-attrs/parallel_tensor_dims.h"
#include "op-attrs/parallel_tensor_shape.h"
#include "op-attrs/tensor_dims.h"
#include "utils/containers/product.h"
#include "utils/containers/zip_with_strict.h"

namespace FlexFlow {

TensorShape reshape_get_output_shape(ReshapeAttrs const &attrs,
                                     TensorShape const &input_shape) {
  ASSERT(attrs.shape.data_type == input_shape.data_type);
  ASSERT(get_num_elements(attrs.shape.dims) ==
         get_num_elements(input_shape.dims));

  return attrs.shape;
}

ParallelTensorDimDegrees reshape_get_output_parallel_dim_degrees(
    ReshapeAttrs const &attrs,
    ParallelTensorDimDegrees const &input_dim_degrees) {
  // TODO: this can (and probably should) be weakened in specific cases,
  // such as where the leading dimensions are not being modified
  ASSERT(product(input_dim_degrees.shard_degrees) == 1);

  return input_dim_degrees;
}

ParallelTensorShape
    reshape_get_output_parallel_shape(ReshapeAttrs const &attrs,
                                      ParallelTensorShape const &input_shape) {
  TensorShape unpar =
      reshape_get_output_shape(attrs, get_reduced_shape(input_shape));
  ParallelTensorDimDegrees degrees = reshape_get_output_parallel_dim_degrees(
      attrs, get_parallel_degrees(input_shape));

  return lift_shape_to_parallel_with_degrees(unpar, degrees);
}

} // namespace FlexFlow
