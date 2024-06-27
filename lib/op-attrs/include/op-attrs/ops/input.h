#ifndef _FLEXFLOW_OP_ATTRS_OPS_OP_ATTRS_INPUT_H
#define _FLEXFLOW_OP_ATTRS_OPS_OP_ATTRS_INPUT_H

#include "core.h"
#include "op-attrs/ops/input_attrs.dtg.h"
#include "op-attrs/parallel_tensor_shape.dtg.h"

namespace FlexFlow {

CHECK_VALID_OP_ATTR(InputAttrs);

ParallelTensorShape get_output_shape(InputAttrs const &);

} // namespace FlexFlow

#endif
