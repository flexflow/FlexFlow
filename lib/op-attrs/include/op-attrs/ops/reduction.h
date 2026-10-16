#ifndef _FLEXFLOW_LIB_OP_ATTRS_INCLUDE_OP_ATTRS_OPS_REDUCTION_H
#define _FLEXFLOW_LIB_OP_ATTRS_INCLUDE_OP_ATTRS_OPS_REDUCTION_H

#include "op-attrs/ops/reduction_attrs.dtg.h"
#include "op-attrs/parallel_tensor_shape.dtg.h"

namespace FlexFlow {

ParallelTensorShape
    reduction_get_output_parallel_shape(ReductionAttrs const &attrs,
                                        ParallelTensorShape const &input_shape);

} // namespace FlexFlow

#endif
