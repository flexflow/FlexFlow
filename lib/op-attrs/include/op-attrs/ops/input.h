#ifndef _FLEXFLOW_OP_ATTRS_OPS_OP_ATTRS_INPUT_H
#define _FLEXFLOW_OP_ATTRS_OPS_OP_ATTRS_INPUT_H

#include "op-attrs/operator_space_to_parallel_tensor_space_mapping.dtg.h"
#include "op-attrs/operator_task_space.dtg.h"
#include "op-attrs/ops/input_attrs.dtg.h"
#include "op-attrs/parallel_tensor_shape.dtg.h"
#include "op-attrs/tensor_shape.dtg.h"

namespace FlexFlow {

TensorShape input_get_output_shape(InputAttrs const &);
ParallelTensorShape input_get_output_parallel_shape(InputAttrs const &);

OperatorTaskSpace input_get_operator_task_space(InputAttrs const &);

OperatorSpaceToParallelTensorSpaceMapping
    input_get_operator_to_output_mapping(InputAttrs const &);

} // namespace FlexFlow

#endif
