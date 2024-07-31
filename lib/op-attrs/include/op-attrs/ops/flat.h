#ifndef _FLEXFLOW_FLAT_ATTRS_H
#define _FLEXFLOW_FLAT_ATTRS_H

#include "op-attrs/ops/core.h"
#include "op-attrs/ops/flat_attrs.dtg.h"
#include "op-attrs/parallel_tensor_shape.dtg.h"
#include "op-attrs/tensor_shape.dtg.h"

namespace FlexFlow {

CHECK_VALID_OP_ATTR(FlatAttrs);

TensorShape get_output_shape(FlatAttrs const &,
                             TensorShape const &);
ParallelTensorShape get_output_shape(FlatAttrs const &,
                                     ParallelTensorShape const &);

} // namespace FlexFlow

#endif
