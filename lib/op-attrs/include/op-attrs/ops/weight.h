#ifndef _FLEXFLOW_LIB_OP_ATTRS_INCLUDE_OP_ATTRS_OPS_WEIGHT_H
#define _FLEXFLOW_LIB_OP_ATTRS_INCLUDE_OP_ATTRS_OPS_WEIGHT_H

#include "op-attrs/ops/core.h"
#include "op-attrs/ops/weight_attrs.dtg.h"
#include "op-attrs/parallel_tensor_shape.dtg.h"
#include "op-attrs/tensor_shape.dtg.h"

namespace FlexFlow {

CHECK_VALID_OP_ATTR(WeightAttrs);

TensorShape get_output_shape(WeightAttrs const &);
ParallelTensorShape get_output_parallel_tensor_shape(WeightAttrs const &);

} // namespace FlexFlow

#endif
