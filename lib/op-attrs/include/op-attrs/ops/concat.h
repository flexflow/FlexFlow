#ifndef _FLEXFLOW_LIB_OP_ATTRS_INCLUDE_OP_ATTRS_OPS_CONCAT_H
#define _FLEXFLOW_LIB_OP_ATTRS_INCLUDE_OP_ATTRS_OPS_CONCAT_H

#include "op-attrs/ops/concat_attrs.dtg.h"
#include "op-attrs/ops/core.h"
#include "op-attrs/parallel_tensor_shape.dtg.h"
#include "op-attrs/tensor_shape.dtg.h"

namespace FlexFlow {

CHECK_VALID_OP_ATTR(ConcatAttrs);

tl::expected<TensorShape, std::string>
    get_output_shape(ConcatAttrs const &,
                     std::vector<TensorShape> const &);
tl::expected<ParallelTensorShape, std::string>
    get_output_shape(ConcatAttrs const &,
                     std::vector<ParallelTensorShape> const &);

} // namespace FlexFlow

#endif
