#ifndef _FLEXFLOW_LIB_OP_ATTRS_INCLUDE_OP_ATTRS_OPS_BROADCAST_H
#define _FLEXFLOW_LIB_OP_ATTRS_INCLUDE_OP_ATTRS_OPS_BROADCAST_H

#include "op-attrs/ops/broadcast_attrs.dtg.h"
#include "op-attrs/parallel_tensor_shape.dtg.h"
#include "op-attrs/tensor_shape.dtg.h"
#include "utils/record_formatter.h"

namespace FlexFlow {

TensorShape broadcast_get_output_shape(BroadcastAttrs const &,
                                       TensorShape const &);
ParallelTensorShape
    broadcast_get_output_parallel_shape(BroadcastAttrs const &,
                                        ParallelTensorShape const &);

} // namespace FlexFlow

#endif
