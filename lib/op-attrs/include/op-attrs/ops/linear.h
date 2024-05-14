#ifndef _FLEXFLOW_LINEAR_ATTRS_H
#define _FLEXFLOW_LINEAR_ATTRS_H

#include "op-attrs/ops/core.h"
#include "op-attrs/ops/linear_attrs.dtg.h"
#include "op-attrs/parallel_tensor_shape.dtg.h"
#include "op-attrs/tensor_shape.dtg.h"

namespace FlexFlow {

CHECK_VALID_OP_ATTR(LinearAttrs);

TensorShape get_kernel_shape(LinearAttrs const &attrs, TensorShape const &input);
TensorShape get_bias_shape(LinearAttrs const &attrs, TensorShape const &input);
TensorShape get_output_shape(LinearAttrs const &attrs, TensorShape const &input);

ParallelTensorShape get_kernel_shape(LinearAttrs const &attrs, ParallelTensorShape const &input);
ParallelTensorShape get_bias_shape(LinearAttrs const &attrs, ParallelTensorShape const &input);
ParallelTensorShape get_output_shape(LinearAttrs const &attrs, ParallelTensorShape const &input);

} // namespace FlexFlow

#endif
