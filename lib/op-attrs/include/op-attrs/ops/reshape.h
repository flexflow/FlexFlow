#ifndef _FLEXFLOW_RESHAPE_ATTRS_H
#define _FLEXFLOW_RESHAPE_ATTRS_H

#include "op-attrs/ops/core.h"
#include "op-attrs/ops/reshape_attrs.dtg.h"
#include "op-attrs/parallel_tensor_shape.dtg.h"

namespace FlexFlow {

CHECK_VALID_OP_ATTR(ReshapeAttrs);

TensorShape get_output_shape(ReshapeAttrs const &attrs,
                             TensorShape const &input_shape);
ParallelTensorShape get_output_shape(ReshapeAttrs const &attrs,
                                     ParallelTensorShape const &input_shape);

} // namespace FlexFlow

#endif
