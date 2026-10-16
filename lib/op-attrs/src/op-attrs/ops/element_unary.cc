#include "op-attrs/ops/element_unary.h"
#include "op-attrs/operator_space_to_parallel_tensor_space_mapping.h"
#include "op-attrs/operator_task_space.h"
#include "op-attrs/parallel_tensor_dim_degrees.h"
#include "op-attrs/parallel_tensor_dim_idx_t.h"
#include "op-attrs/parallel_tensor_shape.h"
#include "utils/orthotope/minimal_dim_domain.h"

namespace FlexFlow {

ElementUnaryAttrs make_relu_attrs() {
  return ElementUnaryAttrs{
      /*op_type=*/ElementUnaryOp::RELU,
      /*scalar=*/std::nullopt,
  };
}

TensorShape element_unary_get_output_shape(ElementUnaryAttrs const &attrs,
                                           TensorShape const &input_shape) {
  return input_shape;
}

ParallelTensorShape element_unary_get_output_parallel_shape(
    ElementUnaryAttrs const &attrs, ParallelTensorShape const &input_shape) {
  TensorShape output_shape =
      element_unary_get_output_shape(attrs, get_reduced_shape(input_shape));

  ParallelTensorDimDegrees output_degrees =
      element_unary_get_output_parallel_dim_degrees(
          attrs, get_parallel_degrees(input_shape));

  return lift_shape_to_parallel_with_degrees(output_shape, output_degrees);
}

ParallelTensorDimDegrees element_unary_get_output_parallel_dim_degrees(
    ElementUnaryAttrs const &attrs,
    ParallelTensorDimDegrees const &input_degrees) {
  ASSERT(input_degrees.sum_degree.value == 1);

  return input_degrees;
}

OperatorTaskSpace element_unary_get_operator_task_space(
    ElementUnaryAttrs const &attrs,
    ParallelTensorDimDegrees const &input_degrees) {
  ParallelTensorDimDegrees output_degrees =
      element_unary_get_output_parallel_dim_degrees(attrs, input_degrees);

  return get_operator_task_space_matching_parallel_tensor_dim_degrees(
      output_degrees);
}

OperatorSpaceToParallelTensorSpaceMapping
    element_unary_get_operator_to_input_mapping(
        ElementUnaryAttrs const &attrs,
        ParallelTensorDimDegrees const &input_degrees) {

  return get_identity_mapping(
      element_unary_get_operator_task_space(attrs, input_degrees),
      input_degrees);
}

OperatorSpaceToParallelTensorSpaceMapping
    element_unary_get_operator_to_output_mapping(
        ElementUnaryAttrs const &attrs,
        ParallelTensorDimDegrees const &input_degrees) {

  ParallelTensorDimDegrees output_degrees =
      element_unary_get_output_parallel_dim_degrees(attrs, input_degrees);

  return get_identity_mapping(
      element_unary_get_operator_task_space(attrs, input_degrees),
      output_degrees);
}

} // namespace FlexFlow
