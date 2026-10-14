#include "op-attrs/ops/weight.h"
#include "op-attrs/operator_space_to_parallel_tensor_space_mapping.h"
#include "op-attrs/operator_task_space.h"
#include "op-attrs/parallel_tensor_shape.h"

namespace FlexFlow {

TensorShape get_output_shape(WeightAttrs const &attrs) {
  return attrs.tensor_shape;
}

ParallelTensorShape get_output_parallel_tensor_shape(WeightAttrs const &attrs) {
  return lift_shape_to_parallel(attrs.tensor_shape);
}

OperatorTaskSpace get_operator_task_space(WeightAttrs const &) {
  return trivial_op_task_space();
}

OperatorSpaceToParallelTensorSpaceMapping
    get_operator_to_output_mapping(WeightAttrs const &attrs) {

  return empty_operator_space_to_ptensor_space_map();
}

} // namespace FlexFlow
