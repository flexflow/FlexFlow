#ifndef _FLEXFLOW_DROPOUT_ATTRS_H
#define _FLEXFLOW_DROPOUT_ATTRS_H

#include "op-attrs/ops/core.h"
#include "op-attrs/ops/dropout_attrs.dtg.h"
#include "op-attrs/parallel_tensor_shape.dtg.h"
#include "op-attrs/tensor_shape.dtg.h"

namespace FlexFlow {

TensorShape get_output_shape(DropoutAttrs const &,
                             TensorShape const &);
ParallelTensorShape get_output_shape(DropoutAttrs const &,
                                     ParallelTensorShape const &);

CHECK_VALID_OP_ATTR(DropoutAttrs);

} // namespace FlexFlow

#endif
