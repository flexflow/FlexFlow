#ifndef _FLEXFLOW_LIB_OP_ATTRS_INCLUDE_OP_ATTRS_OPS_DROPOUT_H
#define _FLEXFLOW_LIB_OP_ATTRS_INCLUDE_OP_ATTRS_OPS_DROPOUT_H

#include "op-attrs/ops/dropout_attrs.dtg.h"
#include "op-attrs/parallel_tensor_shape.dtg.h"
#include "op-attrs/tensor_shape.dtg.h"

namespace FlexFlow {

TensorShape get_output_shape(DropoutAttrs const &, TensorShape const &);
ParallelTensorShape get_output_shape(DropoutAttrs const &,
                                     ParallelTensorShape const &);

} // namespace FlexFlow

#endif
