#include "op-attrs/ops/element_binary.h"
#include "op-attrs/operator_space_to_parallel_tensor_space_mapping.h"
#include "op-attrs/operator_task_space.h"
#include "utils/containers/require_same.h"
#include "utils/exception.h"

namespace FlexFlow {

TensorShape get_output_shape(ElementBinaryAttrs const &attrs,
                             TensorShape const &input_lhs,
                             TensorShape const &input_rhs) {
  ASSERT(!attrs.should_broadcast_lhs && !attrs.should_broadcast_rhs,
         "ElementBinary broadcasting is currently not supported. "
         "Contact @lockshaw if you want this feature implemented.");

  if (attrs.should_broadcast_lhs) {
    NOT_IMPLEMENTED();
  } else if (attrs.should_broadcast_rhs) {
    NOT_IMPLEMENTED();
  } else {
    ASSERT(input_lhs == input_rhs, "Expected input shapes to match");

    return input_lhs;
  }
}

ParallelTensorShape get_output_shape(ElementBinaryAttrs const &attrs,
                                     ParallelTensorShape const &input_lhs,
                                     ParallelTensorShape const &input_rhs) {
  TensorShape output_shape = get_output_shape(
      attrs, get_reduced_shape(input_lhs), get_reduced_shape(input_rhs));

  ParallelTensorDimDegrees output_degrees = get_output_parallel_dim_degrees(
      attrs, get_parallel_degrees(input_lhs), get_parallel_degrees(input_rhs));

  return lift_to_parallel_with_degrees(output_shape, output_degrees);
}

ParallelTensorDimDegrees get_output_parallel_dim_degrees(
    ElementBinaryAttrs const &attrs,
    ParallelTensorDimDegrees const &lhs_input_degrees,
    ParallelTensorDimDegrees const &rhs_input_degrees) {
  ASSERT(!attrs.should_broadcast_lhs && !attrs.should_broadcast_rhs,
         "ElementBinary broadcasting is currently not supported. "
         "Contact @lockshaw if you want this feature implemented.");

  ASSERT(lhs_input_degrees == rhs_input_degrees);

  if (attrs.should_broadcast_lhs) {
    NOT_IMPLEMENTED();
  } else if (attrs.should_broadcast_rhs) {
    NOT_IMPLEMENTED();
  } else {
    ASSERT(lhs_input_degrees == rhs_input_degrees,
           "Expected input degrees to match");

    switch (attrs.op) {
      case ElementBinaryOp::ADD: {
        ASSERT(
            lhs_input_degrees.discard_copy_degree.value == 1,
            "Elementwise Add expected discard copy degree of inputs to be 1");

        break;
      }
      case ElementBinaryOp::SUBTRACT:
        NOT_IMPLEMENTED();
      case ElementBinaryOp::MULTIPLY:
        NOT_IMPLEMENTED();
      case ElementBinaryOp::DIVIDE:
        NOT_IMPLEMENTED();
      case ElementBinaryOp::MAX:
        NOT_IMPLEMENTED();
      case ElementBinaryOp::MIN:
        NOT_IMPLEMENTED();
      default:
        PANIC("Unexpected element-wise binary operator", attrs.op);
    }

    return lhs_input_degrees;
  }
}

OperatorTaskSpace
    get_operator_task_space(ElementBinaryAttrs const &attrs,
                            ParallelTensorDimDegrees const &lhs_input_degrees,
                            ParallelTensorDimDegrees const &rhs_input_degrees) {

  ParallelTensorDimDegrees output_degrees = get_output_parallel_dim_degrees(
      attrs, lhs_input_degrees, rhs_input_degrees);

  return get_operator_task_space_matching_parallel_tensor_dim_degrees(
      output_degrees);
}

OperatorSpaceToParallelTensorSpaceMapping get_operator_to_lhs_input_mapping(
    ElementBinaryAttrs const &attrs,
    ParallelTensorDimDegrees const &lhs_input_degrees,
    ParallelTensorDimDegrees const &rhs_input_degrees) {

  return get_identity_mapping(
      get_operator_task_space(attrs, lhs_input_degrees, rhs_input_degrees),
      lhs_input_degrees);
}

OperatorSpaceToParallelTensorSpaceMapping get_operator_to_rhs_input_mapping(
    ElementBinaryAttrs const &attrs,
    ParallelTensorDimDegrees const &lhs_input_degrees,
    ParallelTensorDimDegrees const &rhs_input_degrees) {

  return get_identity_mapping(
      get_operator_task_space(attrs, lhs_input_degrees, rhs_input_degrees),
      rhs_input_degrees);
}

OperatorSpaceToParallelTensorSpaceMapping get_operator_to_output_mapping(
    ElementBinaryAttrs const &attrs,
    ParallelTensorDimDegrees const &lhs_input_degrees,
    ParallelTensorDimDegrees const &rhs_input_degrees) {

  ParallelTensorDimDegrees output_dim_degrees = get_output_parallel_dim_degrees(
      attrs, lhs_input_degrees, rhs_input_degrees);

  return get_identity_mapping(
      get_operator_task_space(attrs, lhs_input_degrees, rhs_input_degrees),
      output_dim_degrees);
}

} // namespace FlexFlow
