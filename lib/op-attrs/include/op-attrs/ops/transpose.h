#ifndef _FLEXFLOW_OP_META_OPS_TRANSPOSE_ATTRS_H
#define _FLEXFLOW_OP_META_OPS_TRANSPOSE_ATTRS_H

#include "op-attrs/ops/core.h"
#include "op-attrs/ops/transpose_attrs.dtg.h"
#include "op-attrs/parallel_tensor_shape.dtg.h"
#include "op-attrs/tensor_shape.dtg.h"

namespace FlexFlow {

CHECK_VALID_OP_ATTR(TransposeAttrs);

TensorShape get_output_shape(TransposeAttrs const &, TensorShape const &);
ParallelTensorShape get_output_shape(TransposeAttrs const &,
                                     ParallelTensorShape const &);

} // namespace FlexFlow

#endif
