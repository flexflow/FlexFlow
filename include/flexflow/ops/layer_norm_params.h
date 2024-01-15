#pragma once

#include "flexflow/ffconst.h"
#include "flexflow/fftype.h"
#include "flexflow/parallel_tensor.h"

namespace FlexFlow {

struct LayerNormParams {
  LayerID layer_guid;
  std::vector<int> axes;
  bool elementwise_affine;
  float eps;
  bool use_bias;
  char name[MAX_OPNAME];
  bool is_valid(ParallelTensorShape const &) const;
};

bool operator==(LayerNormParams const &, LayerNormParams const &);

} // namespace FlexFlow

namespace std {
template <>
struct hash<FlexFlow::LayerNormParams> {
  size_t operator()(FlexFlow::LayerNormParams const &) const;
};
} // namespace std
