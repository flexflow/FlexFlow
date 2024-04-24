#ifndef _FLEXFLOW_RESHAPE_ATTRS_H
#define _FLEXFLOW_RESHAPE_ATTRS_H

#include "core.h"
#include "op-attrs/ops/reshape_attrs.dtg.h"
#include "op-attrs/parallel_tensor_shape.dtg.h"

namespace FlexFlow {

CHECK_VALID_OP_ATTR(ReshapeAttrs);

ParallelTensorShape get_output_shape(ReshapeAttrs const &attrs, ParallelTensorShape const &input_shape);

} // namespace FlexFlow

#endif
