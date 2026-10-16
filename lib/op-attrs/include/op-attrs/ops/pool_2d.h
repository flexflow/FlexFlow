#ifndef _FLEXFLOW_LIB_OP_ATTRS_INCLUDE_OP_ATTRS_OPS_POOL_2D_H
#define _FLEXFLOW_LIB_OP_ATTRS_INCLUDE_OP_ATTRS_OPS_POOL_2D_H

#include "op-attrs/ops/pool_2d_attrs.dtg.h"
#include "op-attrs/parallel_tensor_dim_degrees.dtg.h"
#include "op-attrs/parallel_tensor_shape.dtg.h"
#include "op-attrs/tensor_shape.dtg.h"

namespace FlexFlow {

Pool2DAttrs
    make_adaptive_pool2d_attrs(TensorDims const &input_dims,
                               positive_int output_h,
                               positive_int output_w,
                               PoolOp pool_type,
                               std::optional<Activation> const &activation);

TensorShape get_output_shape(Pool2DAttrs const &, TensorShape const &);

ParallelTensorShape get_output_shape(Pool2DAttrs const &,
                                     ParallelTensorShape const &);

ParallelTensorDimDegrees
    get_output_parallel_dim_degrees(Pool2DAttrs const &,
                                    ParallelTensorDimDegrees const &);

} // namespace FlexFlow

#endif
