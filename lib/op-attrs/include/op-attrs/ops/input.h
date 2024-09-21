#ifndef _FLEXFLOW_OP_ATTRS_OPS_OP_ATTRS_INPUT_H
#define _FLEXFLOW_OP_ATTRS_OPS_OP_ATTRS_INPUT_H

#include "op-attrs/ops/core.h"
#include "op-attrs/ops/input_attrs.dtg.h"
#include "op-attrs/parallel_tensor_shape.dtg.h"
#include "op-attrs/tensor_shape.dtg.h"

namespace FlexFlow {

CHECK_VALID_OP_ATTR(InputAttrs);

TensorShape get_output_shape(InputAttrs const &);
ParallelTensorShape get_output_parallel_tensor_shape(InputAttrs const &);

} // namespace FlexFlow

#endif
