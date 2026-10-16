#include "op-attrs/ops/input.h"
#include "op-attrs/operator_space_to_parallel_tensor_space_mapping.h"
#include "op-attrs/operator_task_space.h"
#include "op-attrs/parallel_tensor_shape.h"

namespace FlexFlow {

TensorShape input_get_output_shape(InputAttrs const &attrs) {
  return attrs.tensor_shape;
}

ParallelTensorShape input_get_output_parallel_shape(InputAttrs const &attrs) {
  return lift_shape_to_parallel(attrs.tensor_shape);
}

OperatorTaskSpace input_get_operator_task_space(InputAttrs const &) {
  return trivial_op_task_space();
}

OperatorSpaceToParallelTensorSpaceMapping
    input_get_operator_to_output_mapping(InputAttrs const &attrs) {

  return empty_operator_space_to_ptensor_space_map();
}

} // namespace FlexFlow
