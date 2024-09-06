#ifndef _FLEXFLOW_POOL_2D_ATTRS_H
#define _FLEXFLOW_POOL_2D_ATTRS_H

#include "op-attrs/ops/core.h"
#include "op-attrs/ops/pool_2d_attrs.dtg.h"
#include "op-attrs/parallel_tensor_shape.dtg.h"
#include "op-attrs/tensor_shape.dtg.h"

namespace FlexFlow {

CHECK_VALID_OP_ATTR(Pool2DAttrs);

TensorShape get_output_shape(Pool2DAttrs const &, TensorShape const &);
ParallelTensorShape get_output_shape(Pool2DAttrs const &,
                                     ParallelTensorShape const &);

} // namespace FlexFlow

#endif
