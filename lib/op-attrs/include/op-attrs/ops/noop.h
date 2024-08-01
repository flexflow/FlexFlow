#ifndef _FLEXFLOW_OP_ATTRS_OPS_NOOP_H
#define _FLEXFLOW_OP_ATTRS_OPS_NOOP_H

#include "op-attrs/ops/core.h"
#include "op-attrs/ops/noop_attrs.dtg.h"
#include "op-attrs/parallel_tensor_shape.dtg.h"
#include "op-attrs/tensor_shape.dtg.h"

namespace FlexFlow {

CHECK_VALID_OP_ATTR(NoopAttrs);

TensorShape get_output_shape(NoopAttrs const &,
                             TensorShape const &);
ParallelTensorShape get_output_shape(NoopAttrs const &,
                                     ParallelTensorShape const &);

} // namespace FlexFlow

#endif
