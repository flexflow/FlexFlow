#ifndef _FLEXFLOW_PCG_INCLUDE_PCG_FILE_FORMAT_V1_OPS_LAYER_NORM_ATTRS_H
#define _FLEXFLOW_PCG_INCLUDE_PCG_FILE_FORMAT_V1_OPS_LAYER_NORM_ATTRS_H

#include "op-attrs/ops/layer_norm.h"
#include "pcg/file_format/v1/ff_dim.h"
#include "utils/visitable.h"

namespace FlexFlow {

struct V1LayerNormAttrs {
  // The size of this vector must be <= MAX_TENSOR_DIMS
  std::vector<int> axes;
  req<bool> elementwise_affine;
  req<float> eps;
};
FF_VISITABLE_STRUCT(V1LayerNormAttrs, axes, elementwise_affine, eps);
CHECK_IS_JSONABLE(V1LayerNormAttrs);

V1LayerNormAttrs to_v1(LayerNormAttrs const &a);
LayerNormAttrs from_v1(V1LayerNormAttrs const &va);

} // namespace FlexFlow

#endif
