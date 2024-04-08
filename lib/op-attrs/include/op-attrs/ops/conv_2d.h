#ifndef _FLEXFLOW_CONV_2D_ATTRS_H
#define _FLEXFLOW_CONV_2D_ATTRS_H

#include "core.h"
#include "op-attrs/parallel_tensor_shape.h"
#include "op-attrs/tensor_shape.h"
#include "op-attrs/ops/conv_2d_attrs.h"

namespace FlexFlow {

CHECK_VALID_OP_ATTR(Conv2DAttrs);

TensorShape get_kernel_shape(Conv2DAttrs const &, TensorShape const &);
TensorShape get_bias_shape(Conv2DAttrs const &, TensorShape const &);

} // namespace FlexFlow

#endif
