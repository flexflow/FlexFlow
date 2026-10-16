#ifndef _FLEXFLOW_LIB_OP_ATTRS_INCLUDE_OP_ATTRS_OPS_ELEMENT_UNARY_H
#define _FLEXFLOW_LIB_OP_ATTRS_INCLUDE_OP_ATTRS_OPS_ELEMENT_UNARY_H

#include "op-attrs/operator_space_to_parallel_tensor_space_mapping.dtg.h"
#include "op-attrs/operator_task_space.dtg.h"
#include "op-attrs/ops/element_unary_attrs.dtg.h"
#include "op-attrs/parallel_tensor_dim_degrees.dtg.h"
#include "op-attrs/parallel_tensor_shape.dtg.h"
#include "op-attrs/tensor_shape.dtg.h"

namespace FlexFlow {

ElementUnaryAttrs make_relu_attrs();

TensorShape get_output_shape(ElementUnaryAttrs const &, TensorShape const &);
ParallelTensorShape get_output_shape(ElementUnaryAttrs const &,
                                     ParallelTensorShape const &);

ParallelTensorDimDegrees get_output_parallel_dim_degrees(
    ElementUnaryAttrs const &attrs,
    ParallelTensorDimDegrees const &input_degrees);

OperatorTaskSpace
    get_operator_task_space(ElementUnaryAttrs const &attrs,
                            ParallelTensorDimDegrees const &input_degrees);

OperatorSpaceToParallelTensorSpaceMapping get_operator_to_input_mapping(
    ElementUnaryAttrs const &attrs,
    ParallelTensorDimDegrees const &input_degrees);

OperatorSpaceToParallelTensorSpaceMapping get_operator_to_output_mapping(
    ElementUnaryAttrs const &attrs,
    ParallelTensorDimDegrees const &input_degrees);

} // namespace FlexFlow

#endif
