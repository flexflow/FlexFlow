#ifndef _FLEXFLOW_LIB_OP_ATTRS_INCLUDE_OP_ATTRS_OPS_CONV_2D_CONV_2D_INPUT_SHAPE_H
#define _FLEXFLOW_LIB_OP_ATTRS_INCLUDE_OP_ATTRS_OPS_CONV_2D_CONV_2D_INPUT_SHAPE_H

#include "op-attrs/ops/conv_2d/conv_2d_input_shape.dtg.h"
#include "op-attrs/tensor_shape.dtg.h"

namespace FlexFlow {

Conv2DInputShape parse_input_shape(TensorShape const &input);

} // namespace FlexFlow

#endif
