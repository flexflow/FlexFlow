#ifndef _FLEXFLOW_SOFTMAX_ATTRS_H
#define _FLEXFLOW_SOFTMAX_ATTRS_H

#include "op-attrs/ops/core.h"
#include "op-attrs/ops/softmax_attrs.dtg.h"
#include "op-attrs/parallel_tensor_shape.dtg.h"
#include "op-attrs/tensor_shape.dtg.h"

namespace FlexFlow {

CHECK_VALID_OP_ATTR(SoftmaxAttrs);

TensorShape get_output_shape(SoftmaxAttrs const &attrs,
                             TensorShape const &input_shape);
ParallelTensorShape get_output_shape(SoftmaxAttrs const &attrs,
                                     ParallelTensorShape const &input_shape);

} // namespace FlexFlow

#endif
