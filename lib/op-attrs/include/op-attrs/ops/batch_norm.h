#ifndef _FLEXFLOW_OP_ATTRS_INCLUDE_OP_ATTRS_OPS_BATCH_NORM_H
#define _FLEXFLOW_OP_ATTRS_INCLUDE_OP_ATTRS_OPS_BATCH_NORM_H

#include "core.h"
#include "op-attrs/ops/batch_norm_attrs.dtg.h"
#include "op-attrs/parallel_tensor_shape.h"

namespace FlexFlow {

ParallelTensorShape get_output_shape(BatchNormAttrs const &, ParallelTensorShape const &);

CHECK_VALID_OP_ATTR(BatchNormAttrs);

} // namespace FlexFlow

#endif
