#ifndef _FLEXFLOW_LIB_OP_ATTRS_INCLUDE_OP_ATTRS_OPS_SPLIT_H
#define _FLEXFLOW_LIB_OP_ATTRS_INCLUDE_OP_ATTRS_OPS_SPLIT_H

#include "op-attrs/ops/split_attrs.dtg.h"
#include "op-attrs/parallel_tensor_dim_degrees.dtg.h"
#include "op-attrs/parallel_tensor_shape.dtg.h"
#include "op-attrs/tensor_shape.dtg.h"
#include "op-attrs/tensor_slot_name.dtg.h"
#include <vector>

namespace FlexFlow {

std::vector<TensorShape> split_get_output_shapes(SplitAttrs const &,
                                                 TensorShape const &);

std::vector<ParallelTensorDimDegrees> split_get_output_parallel_dim_degrees(
    SplitAttrs const &attrs, ParallelTensorDimDegrees const &input);

std::vector<ParallelTensorShape>
    split_get_output_parallel_shapes(SplitAttrs const &attrs,
                                     ParallelTensorShape const &input_shape);

std::vector<TensorSlotName> split_get_output_slot_names(SplitAttrs const &);

} // namespace FlexFlow

#endif
