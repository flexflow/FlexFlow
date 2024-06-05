#ifndef _FLEXFLOW_LIB_OP_ATTRS_INCLUDE_OP_ATTRS_OPS_COMBINE_H
#define _FLEXFLOW_LIB_OP_ATTRS_INCLUDE_OP_ATTRS_OPS_COMBINE_H

#include "op-attrs/ops/combine_attrs.dtg.h"
#include "op-attrs/ops/core.h"
#include "op-attrs/parallel_tensor_shape.dtg.h"
#include <tl/expected.hpp>

namespace FlexFlow {

CHECK_VALID_OP_ATTR(CombineAttrs);

tl::expected<ParallelTensorShape, std::string>
    get_output_shape(CombineAttrs const &, ParallelTensorShape const &);

} // namespace FlexFlow

#endif
