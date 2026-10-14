#include "op-attrs/ops/split.h"
#include "op-attrs/parallel_tensor_dim_degrees.h"
#include "op-attrs/parallel_tensor_dim_idx_t.h"
#include "op-attrs/parallel_tensor_dims.h"
#include "op-attrs/parallel_tensor_shape.h"
#include "op-attrs/tensor_dims.h"
#include "utils/containers/repeat_element.h"
#include "utils/containers/sum.h"
#include "utils/containers/transform.h"
#include "utils/containers/vector_of.h"
#include "utils/containers/zip_with_strict.h"
#include "utils/nonnegative_int/num_elements.h"

namespace FlexFlow {

std::vector<TensorShape> get_output_shapes(SplitAttrs const &attrs,
                                           TensorShape const &input_shape) {
  ASSERT(sum(attrs.splits) == dim_at_idx(input_shape.dims, attrs.axis));

  auto for_split_size = [&](positive_int split_size) -> TensorShape {
    TensorShape result = input_shape;
    dim_at_idx(result.dims, attrs.axis) = split_size;
    return result;
  };

  return transform(vector_of(attrs.splits), for_split_size);
}

std::vector<ParallelTensorDimDegrees> get_output_parallel_dim_degrees(
    SplitAttrs const &attrs,
    ParallelTensorDimDegrees const &input_dim_degrees) {
  {
    positive_int axis_degree = get_degree_for_parallel_tensor_dim_idx(
        input_dim_degrees, shard_dim_idx(attrs.axis));
    ASSERT(axis_degree == 1_p);
  }

  return repeat_element(num_elements(attrs.splits), input_dim_degrees);
}

std::vector<ParallelTensorShape>
    get_output_shapes(SplitAttrs const &attrs,
                      ParallelTensorShape const &input_shape) {
  std::vector<TensorShape> unpar =
      get_output_shapes(attrs, get_reduced_shape(input_shape));
  std::vector<ParallelTensorDimDegrees> degrees =
      get_output_parallel_dim_degrees(attrs, get_parallel_degrees(input_shape));

  return zip_with_strict(
      unpar,
      degrees,
      [&](TensorShape const &s,
          ParallelTensorDimDegrees const &d) -> ParallelTensorShape {
        return lift_to_parallel_with_degrees(s, d);
      });
}

} // namespace FlexFlow
