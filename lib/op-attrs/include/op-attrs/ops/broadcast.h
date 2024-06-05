#ifndef _FLEXFLOW_LIB_OP_ATTRS_INCLUDE_OP_ATTRS_OPS_BROADCAST_H
#define _FLEXFLOW_LIB_OP_ATTRS_INCLUDE_OP_ATTRS_OPS_BROADCAST_H

#include "op-attrs/ops/broadcast.dtg.h"
#include "op-attrs/parallel_tensor_shape.dtg.h"

namespace FlexFlow {

ParallelTensorShape get_output_shape(BroadcastAttrs const &,
                                     ParallelTensorShape const &);

} // namespace FlexFlow

#endif
