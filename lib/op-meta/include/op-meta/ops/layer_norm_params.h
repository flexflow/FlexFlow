#ifndef _FLEXFLOW_OP_META_OPS_LAYER_NORM_PARAMS_H
#define _FLEXFLOW_OP_META_OPS_LAYER_NORM_PARAMS_H

#include "op-meta/parallel_tensor_shape.h"

namespace FlexFlow {

struct LayerNormParams {
  std::vector<int> axes;
  bool elementwise_affine;
  float eps;
  bool is_valid(ParallelTensorShape const &) const;
};

bool operator==(LayerNormParams const &, LayerNormParams const &);

} 

namespace std {
template <>
struct hash<FlexFlow::LayerNormParams> {
  size_t operator()(FlexFlow::LayerNormParams const &) const;
};
}

#endif // _FLEXFLOW_OP_META_OPS_LAYER_NORM_PARAMS_H
