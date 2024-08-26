#ifndef _FLEXFLOW_CONCAT_ATTRS_H
#define _FLEXFLOW_CONCAT_ATTRS_H

#include "op-attrs/ops/concat_attrs.dtg.h"
#include "op-attrs/ops/core.h"
#include "op-attrs/parallel_tensor_shape.dtg.h"
#include "op-attrs/tensor_shape.dtg.h"

namespace FlexFlow {

CHECK_VALID_OP_ATTR(ConcatAttrs);

TensorShape get_output_shape(ConcatAttrs const &,
                             std::vector<TensorShape> const &);
ParallelTensorShape get_output_shape(ConcatAttrs const &,
                                     std::vector<ParallelTensorShape> const &);

} // namespace FlexFlow

#endif
