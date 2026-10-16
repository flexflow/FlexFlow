#ifndef _FLEXFLOW_LIB_OP_ATTRS_INCLUDE_OP_ATTRS_OPS_TOPK_H
#define _FLEXFLOW_LIB_OP_ATTRS_INCLUDE_OP_ATTRS_OPS_TOPK_H

#include "op-attrs/ops/topk_attrs.dtg.h"
#include "op-attrs/parallel_tensor_shape.dtg.h"
#include "op-attrs/tensor_shape.dtg.h"

namespace FlexFlow {

TensorShape topk_get_output_shape(TopKAttrs const &, TensorShape const &);
ParallelTensorShape
    topk_get_output_parallel_shape(TopKAttrs const &attrs,
                                   ParallelTensorShape const &input_shape);

} // namespace FlexFlow

#endif
