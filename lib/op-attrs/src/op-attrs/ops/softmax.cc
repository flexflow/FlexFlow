#include "op-attrs/ops/softmax.h"
#include "op-attrs/parallel_tensor_shape.h"
#include "op-attrs/tensor_shape.h"

namespace FlexFlow {

tl::expected<TensorShape, std::string>
    get_output_shape(SoftmaxAttrs const &attrs,
                     TensorShape const &input_shape) {
  if (attrs.dim.value >= num_dims(input_shape)) {
    return tl::unexpected(
        fmt::format("get_output_shape for Softmax received out-of-bounds "
                    "attrs.dim {} for input tensor shape {}",
                    attrs.dim,
                    input_shape));
  }

  return input_shape;
}

tl::expected<ParallelTensorShape, std::string>
    get_output_shape(SoftmaxAttrs const &attrs,
                     ParallelTensorShape const &input_shape) {
  tl::expected<TensorShape, std::string> result_unpar =
      get_output_shape(attrs, get_reduced_shape(input_shape));
  if (!result_unpar.has_value()) {
    return tl::unexpected(result_unpar.error());
  }

  if (get_sum_degree(input_shape) != 1) {
    return tl::unexpected(
        fmt::format("Expected sum degree 1, but receieved sum degree {}",
                    get_sum_degree(input_shape)));
  }

  if (get_discard_copy_degree(input_shape) != 1) {
    return tl::unexpected(fmt::format(
        "Expected discard copy degree 1, but received discard copy degree {}",
        get_discard_copy_degree(input_shape)));
  }

  if (shard_dim_at_idx(input_shape, attrs.dim).degree != 1) {
    return tl::unexpected(
        fmt::format("Expected parallel degree of Softmax dimension {} to be 1, "
                    "but received input shape {}",
                    attrs.dim,
                    input_shape));
  }

  return input_shape;
}

} // namespace FlexFlow
