#ifndef _FLEXFLOW_FLAT_ATTRS_H
#define _FLEXFLOW_FLAT_ATTRS_H

#include "op-attrs/ops/core.h"
#include "op-attrs/ops/flat_attrs.dtg.h"
#include "op-attrs/parallel_tensor_dim_degrees.dtg.h"
#include "op-attrs/parallel_tensor_shape.dtg.h"
#include "op-attrs/tensor_shape.dtg.h"

namespace FlexFlow {

CHECK_VALID_OP_ATTR(FlatAttrs);

TensorShape get_output_shape(FlatAttrs const &, TensorShape const &);
tl::expected<ParallelTensorDimDegrees, std::string>
    get_output_parallel_dim_degrees(FlatAttrs const &,
                                    ParallelTensorDimDegrees const &);
tl::expected<ParallelTensorShape, std::string>
    get_output_shape(FlatAttrs const &, ParallelTensorShape const &);

} // namespace FlexFlow

#endif
