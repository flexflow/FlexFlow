#ifndef _FLEXFLOW_GATHER_ATTRS_H
#define _FLEXFLOW_GATHER_ATTRS_H

#include "core.h"
#include "op-attrs/ops/gather_attrs.dtg.h"
#include "op-attrs/parallel_tensor_shape.h"

namespace FlexFlow {

CHECK_VALID_OP_ATTR(GatherAttrs);

TensorShape get_output_shape(GatherAttrs const &,
                             TensorShape const &input,
                             TensorShape const &index);
ParallelTensorShape get_output_shape(GatherAttrs const &,
                                     ParallelTensorShape const &input,
                                     ParallelTensorShape const &index);

} // namespace FlexFlow

#endif
