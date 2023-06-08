#ifndef _FLEXFLOW_OP_META_OPS_LAYER_NORM_ATTRS_H
#define _FLEXFLOW_OP_META_OPS_LAYER_NORM_ATTRS_H

#include "op-attrs/ff_dim.h"
#include "op-attrs/parallel_tensor_shape.h"
#include "utils/visitable.h"

namespace FlexFlow {

struct LayerNormAttrs : use_visitable_cmp<LayerNormAttrs> {
public:
  LayerNormAttrs(stack_vector<ff_dim_t, MAX_TENSOR_DIM> const &axes,
                 bool elementwise_affine, float eps);

public:
  stack_vector<ff_dim_t, MAX_TENSOR_DIM> axes;
  bool elementwise_affine;
  float eps;
};

} // namespace FlexFlow

VISITABLE_STRUCT(::FlexFlow::LayerNormAttrs, axes, elementwise_affine, eps);
MAKE_VISIT_HASHABLE(::FlexFlow::LayerNormAttrs);

#endif
