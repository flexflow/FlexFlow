#ifndef _FLEXFLOW_LIB_OP_ATTRS_INCLUDE_OP_ATTRS_OPS_LINEAR_H
#define _FLEXFLOW_LIB_OP_ATTRS_INCLUDE_OP_ATTRS_OPS_LINEAR_H

#include "op-attrs/incoming_tensor_role.dtg.h"
#include "op-attrs/initializer_attrs.dtg.h"
#include "op-attrs/num_ptensor_parallel_dims_t.h"
#include "op-attrs/operator_space_to_parallel_tensor_space_mapping.dtg.h"
#include "op-attrs/operator_task_space.dtg.h"
#include "op-attrs/ops/linear_attrs.dtg.h"
#include "op-attrs/parallel_tensor_dim_degrees.dtg.h"
#include "op-attrs/parallel_tensor_shape.dtg.h"
#include "op-attrs/parallel_tensor_space_to_parallel_tensor_space_mapping.dtg.h"
#include "op-attrs/tensor_shape.dtg.h"
#include "op-attrs/tensor_slot_name.dtg.h"
#include "utils/record_formatter.h"

namespace FlexFlow {

std::map<TensorSlotName, IncomingTensorRole>
    get_linear_incoming_tensor_roles(LinearAttrs const &);

TensorShape get_projection_shape(LinearAttrs const &attrs,
                                 TensorShape const &input);
TensorShape get_bias_shape(LinearAttrs const &attrs, TensorShape const &input);
TensorShape get_output_shape(LinearAttrs const &attrs,
                             TensorShape const &input);

std::map<TensorSlotName, TensorShape>
    get_weight_shapes(LinearAttrs const &attrs, TensorShape const &input_shape);

ParallelTensorDimDegrees
    get_projection_parallel_dim_degrees(LinearAttrs const &attrs,
                                        ParallelTensorDimDegrees const &input);
ParallelTensorDimDegrees
    get_bias_parallel_dim_degrees(LinearAttrs const &attrs,
                                  ParallelTensorDimDegrees const &input);
ParallelTensorDimDegrees
    get_output_parallel_dim_degrees(LinearAttrs const &attrs,
                                    ParallelTensorDimDegrees const &input);

ParallelTensorShape get_projection_shape(LinearAttrs const &attrs,
                                         ParallelTensorShape const &input);
ParallelTensorShape get_bias_shape(LinearAttrs const &attrs,
                                   ParallelTensorShape const &input);
ParallelTensorShape get_output_shape(LinearAttrs const &attrs,
                                     ParallelTensorShape const &input);

std::map<TensorSlotName, ParallelTensorShape>
    get_weight_shapes(LinearAttrs const &attrs,
                      ParallelTensorShape const &input_shape);

std::map<TensorSlotName, InitializerAttrs> get_initializers(
    LinearAttrs const &,
    TensorShape const &input_shape,
    std::optional<InitializerAttrs> const &projection_initializer =
        std::nullopt,
    std::optional<InitializerAttrs> const &kernel_initializer = std::nullopt);

OperatorTaskSpace
    get_operator_task_space(LinearAttrs const &attrs,
                            ParallelTensorDimDegrees const &input_degrees);

OperatorSpaceToParallelTensorSpaceMapping get_operator_to_input_mapping(
    LinearAttrs const &attrs, ParallelTensorDimDegrees const &input_degrees);
OperatorSpaceToParallelTensorSpaceMapping get_operator_to_projection_mapping(
    LinearAttrs const &attrs, ParallelTensorDimDegrees const &input_degrees);
OperatorSpaceToParallelTensorSpaceMapping
    get_operator_to_bias_mapping(LinearAttrs const &attrs,
                                 ParallelTensorDimDegrees const &input_degrees);

OperatorSpaceToParallelTensorSpaceMapping get_operator_to_output_mapping(
    LinearAttrs const &attrs, ParallelTensorDimDegrees const &input_degrees);

} // namespace FlexFlow

#endif
