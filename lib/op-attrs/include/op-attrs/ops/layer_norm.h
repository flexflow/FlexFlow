#ifndef _FLEXFLOW_OP_META_OPS_LAYER_NORM_ATTRS_H
#define _FLEXFLOW_OP_META_OPS_LAYER_NORM_ATTRS_H

#include "core.h"
#include "op-attrs/ff_dim.h"
#include "op-attrs/parallel_tensor_shape.h"
#include "utils/visitable.h"

namespace FlexFlow {

struct LayerNormAttrs {
  stack_vector<ff_dim_t, MAX_TENSOR_DIM> axes;
  req<bool> elementwise_affine;
  req<float> eps;
};
FF_VISITABLE_STRUCT(LayerNormAttrs, axes, elementwise_affine, eps);
CHECK_VALID_OP_ATTR(LayerNormAttrs);

} // namespace FlexFlow

#endif
