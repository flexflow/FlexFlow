#ifndef _FLEXFLOW_OP_META_OPS_LAYER_NORM_PARAMS_H
#define _FLEXFLOW_OP_META_OPS_LAYER_NORM_PARAMS_H

#include "op-meta/parallel_tensor_shape.h"
#include "op-meta/ops/op_params.h"

namespace FlexFlow {

struct LayerNormParams : public OpParamsInterface {
public:
  bool is_valid(ParallelTensorShape const &) const;

  using AsConstTuple = std::tuple<std::vector<int>, bool, float>;
  AsConstTuple as_tuple() const;
public:
  std::vector<int> axes;
  bool elementwise_affine;
  float eps;
};

bool operator==(LayerNormParams const &, LayerNormParams const &);
bool operator<(LayerNormParams const &, LayerNormParams const &);

} 

namespace std {
template <>
struct hash<FlexFlow::LayerNormParams> {
  size_t operator()(FlexFlow::LayerNormParams const &) const;
};
}

#endif // _FLEXFLOW_OP_META_OPS_LAYER_NORM_PARAMS_H
