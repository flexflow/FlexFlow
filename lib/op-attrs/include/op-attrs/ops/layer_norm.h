#ifndef _FLEXFLOW_OP_META_OPS_LAYER_NORM_ATTRS_H
#define _FLEXFLOW_OP_META_OPS_LAYER_NORM_ATTRS_H

#include "op-attrs/ops/core.h"
#include "op-attrs/ops/layer_norm_attrs.dtg.h"
#include "op-attrs/parallel_tensor_shape.dtg.h"
#include "op-attrs/tensor_shape.dtg.h"

namespace FlexFlow {

TensorShape get_output_shape(LayerNormAttrs const &, TensorShape const &);
ParallelTensorShape get_output_shape(LayerNormAttrs const &,
                                     ParallelTensorShape const &);

CHECK_VALID_OP_ATTR(LayerNormAttrs);

} // namespace FlexFlow

#endif
