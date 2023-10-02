#ifndef _FLEXFLOW_PCG_INCLUDE_PCG_FILE_FORMAT_V1_OPS_LAYER_NORM_ATTRS_H
#define _FLEXFLOW_PCG_INCLUDE_PCG_FILE_FORMAT_V1_OPS_LAYER_NORM_ATTRS_H

#include "op-attrs/ff_dim.h"
#include "op-attrs/ops/layer_norm.h"
#include "utils/visitable.h"

namespace FlexFlow {

struct V1LayerNormAttrs {
  // FIXME: stack_vector is probably causing a problem because it is not
  // automatically (de)serializable, but need to ensure that one way or the
  // other.
  // stack_vector<ff_dim_t, MAX_TENSOR_DIM> axes;
  req<bool> elementwise_affine;
  req<float> eps;
};
FF_VISITABLE_STRUCT(V1LayerNormAttrs, // axes,
                    elementwise_affine, eps);
CHECK_IS_JSONABLE(V1LayerNormAttrs);

V1LayerNormAttrs to_v1(LayerNormAttrs const &attrs);

} // namespace FlexFlow

#endif
