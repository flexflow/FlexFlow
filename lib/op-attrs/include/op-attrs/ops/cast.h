#ifndef _FLEXFLOW_LIB_OP_ATTRS_INCLUDE_OP_ATTRS_OPS_CAST_H
#define _FLEXFLOW_LIB_OP_ATTRS_INCLUDE_OP_ATTRS_OPS_CAST_H

#include "op-attrs/ops/cast_attrs.dtg.h"
#include "op-attrs/parallel_tensor_shape.dtg.h"
#include "op-attrs/tensor_shape.dtg.h"
#include "utils/record_formatter.h"

namespace FlexFlow {

TensorShape cast_get_output_shape(CastAttrs const &, TensorShape const &);

ParallelTensorShape cast_get_output_parallel_shape(CastAttrs const &,
                                                   ParallelTensorShape const &);

} // namespace FlexFlow

#endif
