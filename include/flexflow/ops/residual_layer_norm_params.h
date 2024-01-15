#pragma once

#include "flexflow/ffconst.h"
#include "flexflow/fftype.h"
#include "flexflow/parallel_tensor.h"

namespace FlexFlow {

struct ResidualLayerNormParams {
  LayerID layer_guid;
  std::vector<int> axes;
  bool elementwise_affine;
  float eps;
  bool use_bias;
  bool use_two_residuals;
  char name[MAX_OPNAME];
  bool is_valid(std::tuple<ParallelTensorShape,
                           ParallelTensorShape,
                           ParallelTensorShape> const &) const;
};

bool operator==(ResidualLayerNormParams const &,
                ResidualLayerNormParams const &);

} // namespace FlexFlow

namespace std {
template <>
struct hash<FlexFlow::ResidualLayerNormParams> {
  size_t operator()(FlexFlow::ResidualLayerNormParams const &) const;
};
} // namespace std
