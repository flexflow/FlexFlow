#include "op-attrs/ops/transpose.h"
#include "op-attrs/operator_space_to_parallel_tensor_space_mapping.h"
#include "op-attrs/operator_task_space.h"
#include "op-attrs/parallel_tensor_dim_idx_t.h"
#include "op-attrs/parallel_tensor_shape.h"
#include "op-attrs/parallel_tensor_space_to_parallel_tensor_space_mapping.dtg.h"
#include "op-attrs/parallel_tensor_space_to_parallel_tensor_space_mapping.h"
#include "utils/bidict/algorithms/bidict_transform_keys.h"
#include "utils/bidict/algorithms/bidict_transform_values.h"

namespace FlexFlow {

TensorShape get_output_shape(TransposeAttrs const &attrs,
                             TensorShape const &input_shape) {
  return permute_tensor_shape(attrs.permutation, input_shape);
}

ParallelTensorDimDegrees get_output_parallel_dim_degrees(
    TransposeAttrs const &attrs,
    ParallelTensorDimDegrees const &input_degrees) {
  return permute_parallel_tensor_dim_degrees(attrs.permutation, input_degrees);
}

ParallelTensorShape get_output_shape(TransposeAttrs const &attrs,
                                     ParallelTensorShape const &input_shape) {
  TensorShape output_shape =
      get_output_shape(attrs, get_reduced_shape(input_shape));

  ParallelTensorDimDegrees output_degrees =
      get_output_parallel_dim_degrees(attrs, get_parallel_degrees(input_shape));

  return lift_shape_to_parallel_with_degrees(output_shape, output_degrees);
}

OperatorTaskSpace
    get_operator_task_space(TransposeAttrs const &attrs,
                            ParallelTensorDimDegrees const &input_degrees) {
  ParallelTensorDimDegrees output_degrees =
      get_output_parallel_dim_degrees(attrs, input_degrees);

  return get_operator_task_space_matching_parallel_tensor_dim_degrees(
      output_degrees);
}

static ParallelTensorSpaceToParallelTensorSpaceMapping
    get_input_to_output_mapping(TransposeAttrs const &attrs,
                                ParallelTensorDimDegrees const &input_degrees) {
  auto ff_dim_to_pt_dim = [](ff_dim_t d) -> parallel_tensor_dim_idx_t {
    return parallel_tensor_dim_idx_t{d};
  };

  EqProjection<parallel_tensor_dim_idx_t, parallel_tensor_dim_idx_t>
      inp_to_out = EqProjection{
          bidict_transform_keys(
              bidict_transform_values(attrs.permutation.as_bidict(),
                                      ff_dim_to_pt_dim),
              ff_dim_to_pt_dim),
      };

  project_dims(inp_to_out, sum_dim_idx(), sum_dim_idx());
  project_dims(inp_to_out, discard_copy_dim_idx(), discard_copy_dim_idx());

  ParallelTensorDimDegrees output_degrees =
      get_output_parallel_dim_degrees(attrs, input_degrees);

  return parallel_tensor_space_mapping_from_projection(
      DimProjection{inp_to_out}, input_degrees, output_degrees);
}

OperatorSpaceToParallelTensorSpaceMapping get_operator_to_input_mapping(
    TransposeAttrs const &attrs,
    ParallelTensorDimDegrees const &input_degrees) {
  ParallelTensorSpaceToParallelTensorSpaceMapping inp_to_out =
      get_input_to_output_mapping(attrs, input_degrees);

  ParallelTensorSpaceToParallelTensorSpaceMapping out_to_inp =
      invert_parallel_tensor_space_mapping(inp_to_out);

  OperatorSpaceToParallelTensorSpaceMapping op_to_out =
      get_operator_to_output_mapping(attrs, input_degrees);

  return operator_ptensor_space_mapping_from_composition(op_to_out, out_to_inp);
}

OperatorSpaceToParallelTensorSpaceMapping get_operator_to_output_mapping(
    TransposeAttrs const &attrs,
    ParallelTensorDimDegrees const &input_degrees) {
  ParallelTensorDimDegrees output_degrees =
      get_output_parallel_dim_degrees(attrs, input_degrees);

  return get_identity_mapping(get_operator_task_space(attrs, input_degrees),
                              output_degrees);
}

} // namespace FlexFlow
