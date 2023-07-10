#ifndef _FLEXFLOW_OP_ATTRS_INCLUDE_OP_ATTRS_OPS_BATCH_NORM_H
#define _FLEXFLOW_OP_ATTRS_INCLUDE_OP_ATTRS_OPS_BATCH_NORM_H

#include "core.h"
#include "op-attrs/parallel_tensor_shape.h"
#include "utils/visitable.h"

namespace FlexFlow {

struct BatchNormAttrs {
  req<bool> relu;
};
FF_VISITABLE_STRUCT(BatchNormAttrs, relu);

ParallelTensorShape get_output_shape(BatchNormAttrs const &);

CHECK_VALID_OP_ATTR(BatchNormAttrs);

} // namespace FlexFlow

#endif
