#ifndef _FLEXFLOW_REDUCTION_ATTRS_H
#define _FLEXFLOW_REDUCTION_ATTRS_H

#include "core.h"
#include "op-attrs/ops/reduction_attrs.dtg.h"
#include "op-attrs/parallel_tensor_shape.dtg.h"

namespace FlexFlow {

CHECK_VALID_OP_ATTR(ReductionAttrs);

ParallelTensorShape get_output_shape(ReductionAttrs const &attrs, ParallelTensorShape const &input_shape);

} // namespace FlexFlow

#endif
