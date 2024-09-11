#ifndef _FLEXFLOW_CAST_ATTRS_H
#define _FLEXFLOW_CAST_ATTRS_H

#include "op-attrs/ops/cast_attrs.dtg.h"
#include "op-attrs/ops/core.h"
#include "op-attrs/parallel_tensor_shape.dtg.h"
#include "op-attrs/tensor_shape.dtg.h"
#include <tl/expected.hpp>

namespace FlexFlow {

CHECK_VALID_OP_ATTR(CastAttrs);

tl::expected<TensorShape, std::string> get_output_shape(CastAttrs const &,
                                                        TensorShape const &);
tl::expected<ParallelTensorShape, std::string>
    get_output_shape(CastAttrs const &, ParallelTensorShape const &);

} // namespace FlexFlow

#endif
