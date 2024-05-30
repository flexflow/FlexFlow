#include "op-attrs/ops/element_binary.h"

namespace FlexFlow {

tl::expected<TensorShape, std::string>
get_output_shape(ElementBinaryAttrs const &attrs,
                           TensorShape const &input_lhs,
                           TensorShape const &input_rhs) {
  assert (!(attrs.should_broadcast_lhs && attrs.should_broadcast_rhs));

  if (attrs.should_broadcast_lhs) {
    NOT_IMPLEMENTED();    
  } else if (attrs.should_broadcast_rhs) {
    NOT_IMPLEMENTED();
  } else {
    if (input_lhs != input_rhs) {
      return tl::unexpected(fmt::format("Expected input shapes to match, but receieved LHS ({}) != RHS ({})", input_lhs, input_rhs));
    }

    return input_lhs;
  }
}

tl::expected<ParallelTensorShape, std::string>
  get_output_shape(ElementBinaryAttrs const &attrs,
                                     ParallelTensorShape const &input_lhs,
                                     ParallelTensorShape const &input_rhs) {
  assert (!(attrs.should_broadcast_lhs && attrs.should_broadcast_rhs));

  if (attrs.should_broadcast_lhs) {
    NOT_IMPLEMENTED();    
  } else if (attrs.should_broadcast_rhs) {
    NOT_IMPLEMENTED();
  } else {
    if (input_lhs != input_rhs) {
      return tl::unexpected(fmt::format("Expected input shapes to match, but receieved LHS ({}) != RHS ({})", input_lhs, input_rhs));
    }

    switch (attrs.type) {
      case OperatorType::EW_ADD: 
      {
        if (get_discard_copy_degree(input_lhs) != 1) {
          return tl::unexpected(fmt::format("Elementwise Add expected discard copy degree of inputs to be 1, but receieved {}", get_discard_copy_degree(input_lhs)));
        }

        break;
      }
      case OperatorType::EW_SUB:
        NOT_IMPLEMENTED();
      case OperatorType::EW_MUL:
        NOT_IMPLEMENTED();
      case OperatorType::EW_DIV:
        NOT_IMPLEMENTED();
      case OperatorType::EW_MAX:
        NOT_IMPLEMENTED();
      case OperatorType::EW_MIN:
        NOT_IMPLEMENTED();
      default:
        return tl::unexpected(fmt::format("Unexpected element-wise binary operator {}", attrs.type));
    }

    return input_lhs;
  }
}

} // namespace FlexFlow
