#ifndef _FLEXFLOW_POOL_2D_ATTRS_H
#define _FLEXFLOW_POOL_2D_ATTRS_H

#include "op-attrs/ops/core.h"
#include "op-attrs/ops/pool_2d_attrs.dtg.h"
#include "op-attrs/parallel_tensor_shape.dtg.h"
#include "op-attrs/tensor_shape.dtg.h"
#include "op-attrs/parallel_tensor_dim_degrees.dtg.h"

namespace FlexFlow {

CHECK_VALID_OP_ATTR(Pool2DAttrs);

tl::expected<TensorShape, std::string>
  get_output_shape(Pool2DAttrs const &, TensorShape const &);

tl::expected<ParallelTensorShape, std::string>
  get_output_shape(Pool2DAttrs const &, ParallelTensorShape const &);

tl::expected<ParallelTensorDimDegrees, std::string> 
  get_output_parallel_dim_degrees(Pool2DAttrs const &, ParallelTensorDimDegrees const &);

} // namespace FlexFlow

#endif
