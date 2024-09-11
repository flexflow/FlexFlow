#ifndef _FLEXFLOW_SPLIT_ATTRS_H
#define _FLEXFLOW_SPLIT_ATTRS_H

#include "op-attrs/ops/core.h"
#include "op-attrs/ops/split_attrs.dtg.h"
#include "op-attrs/parallel_tensor_shape.dtg.h"
#include "op-attrs/tensor_shape.dtg.h"
#include <vector>

namespace FlexFlow {

CHECK_VALID_OP_ATTR(SplitAttrs);

std::vector<TensorShape> get_output_shapes(SplitAttrs const &,
                                           TensorShape const &);
std::vector<ParallelTensorShape>
    get_output_shapes(SplitAttrs const &attrs,
                      ParallelTensorShape const &input_shape);

} // namespace FlexFlow

#endif
