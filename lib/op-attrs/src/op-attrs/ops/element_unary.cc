#include "op-attrs/ops/element_unary.h"
#include "op-attrs/parallel_tensor_shape.h"

namespace FlexFlow {

tl::expected<TensorShape, std::string>
  get_output_shape(ElementUnaryAttrs const &attrs, TensorShape const &input_shape) {
  return input_shape;
}

tl::expected<ParallelTensorShape, std::string> get_output_shape(ElementUnaryAttrs const &attrs,
                                     ParallelTensorShape const &input_shape) {
  if (get_sum_degree(input_shape) != 1) {
    return tl::unexpected(fmt::format("Expected sum degree 1, but receieved sum degree {}", get_sum_degree(input_shape)));
  }

  if (get_discard_copy_degree(input_shape) != 1) {
    return tl::unexpected(fmt::format("Expected discard copy degree 1, but received discartd copy degree {}", get_discard_copy_degree(input_shape))); 
  }

  return input_shape;
}

tl::expected<TensorShape, std::string>
  get_output_shape(ElementScalarUnaryAttrs const &attrs,
                             TensorShape const &input_shape) {
  return input_shape;
}

tl::expected<ParallelTensorShape, std::string>
  get_output_shape(ElementScalarUnaryAttrs const &attrs,
                                     ParallelTensorShape const &input_shape) {
  if (get_sum_degree(input_shape) != 1) {
    return tl::unexpected(fmt::format("Expected sum degree 1, but receieved sum degree {}", get_sum_degree(input_shape)));
  }

  if (get_discard_copy_degree(input_shape) != 1) {
    return tl::unexpected(fmt::format("Expected discard copy degree 1, but received discartd copy degree {}", get_discard_copy_degree(input_shape))); 
  }

  return input_shape;
}


} // namespace FlexFlow
