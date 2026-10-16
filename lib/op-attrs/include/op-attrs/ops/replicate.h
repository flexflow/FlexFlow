#ifndef _FLEXFLOW_LIB_OP_ATTRS_INCLUDE_OP_ATTRS_OPS_REPLICATE_H
#define _FLEXFLOW_LIB_OP_ATTRS_INCLUDE_OP_ATTRS_OPS_REPLICATE_H

#include "op-attrs/ops/replicate_attrs.dtg.h"
#include "op-attrs/parallel_tensor_shape.dtg.h"

namespace FlexFlow {

ParallelTensorShape
    replicate_get_output_parallel_shape(ReplicateAttrs const &attrs,
                                        ParallelTensorShape const &input_shape);

} // namespace FlexFlow

#endif
