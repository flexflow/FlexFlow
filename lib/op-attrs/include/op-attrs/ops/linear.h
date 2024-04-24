#ifndef _FLEXFLOW_LINEAR_ATTRS_H
#define _FLEXFLOW_LINEAR_ATTRS_H

#include "op-attrs/ops/core.h"
#include "op-attrs/ops/linear_attrs.dtg.h"
#include "op-attrs/parallel_tensor_shape.dtg.h"

namespace FlexFlow {

CHECK_VALID_OP_ATTR(LinearAttrs);

ParallelTensorShape get_output_shape(LinearAttrs const &, ParallelTensorShape const &);

} // namespace FlexFlow

#endif
