#ifndef _FLEXFLOW_LIB_OP_ATTRS_INCLUDE_OP_ATTRS_OPS_SOFTMAX_H
#define _FLEXFLOW_LIB_OP_ATTRS_INCLUDE_OP_ATTRS_OPS_SOFTMAX_H

#include "op-attrs/ops/softmax_attrs.dtg.h"
#include "op-attrs/parallel_tensor_dim_degrees.dtg.h"
#include "op-attrs/parallel_tensor_shape.dtg.h"
#include "op-attrs/tensor_shape.dtg.h"

namespace FlexFlow {

TensorShape softmax_get_output_shape(SoftmaxAttrs const &attrs,
                                     TensorShape const &input_shape);

ParallelTensorDimDegrees softmax_get_output_parallel_dim_degrees(
    SoftmaxAttrs const &attrs, ParallelTensorDimDegrees const &input_degrees);

ParallelTensorShape
    softmax_get_output_parallel_shape(SoftmaxAttrs const &attrs,
                                      ParallelTensorShape const &input_shape);

} // namespace FlexFlow

#endif
