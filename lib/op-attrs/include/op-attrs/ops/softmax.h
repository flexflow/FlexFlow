#ifndef _FLEXFLOW_SOFTMAX_ATTRS_H
#define _FLEXFLOW_SOFTMAX_ATTRS_H

#include "core.h"
#include "op-attrs/ops/softmax_attrs.dtg.h"
#include "op-attrs/parallel_tensor_shape.dtg.h"

namespace FlexFlow {

CHECK_VALID_OP_ATTR(SoftmaxAttrs);

ParallelTensorShape get_output_shape(SoftmaxAttrs const &attrs, ParallelTensorShape const &input_shape);

} // namespace FlexFlow

#endif
