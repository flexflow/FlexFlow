#ifndef _FLEXFLOW_LIB_OP_ATTRS_INCLUDE_OP_ATTRS_OPS_UPSAMPLE_H
#define _FLEXFLOW_LIB_OP_ATTRS_INCLUDE_OP_ATTRS_OPS_UPSAMPLE_H

#include "op-attrs/ops/upsample_attrs.dtg.h"
#include "op-attrs/parallel_tensor_dim_degrees.dtg.h"
#include "op-attrs/parallel_tensor_shape.dtg.h"
#include "op-attrs/tensor_shape.dtg.h"

namespace FlexFlow {

TensorShape upsample_get_output_shape(UpsampleAttrs const &attrs,
                                      TensorShape const &input_shape);

ParallelTensorDimDegrees upsample_get_output_parallel_dim_degrees(
    UpsampleAttrs const &attrs,
    ParallelTensorDimDegrees const &input_dim_degrees);

ParallelTensorShape
    upsample_get_output_parallel_shape(UpsampleAttrs const &attrs,
                                       ParallelTensorShape const &input_shape);

} // namespace FlexFlow

#endif
