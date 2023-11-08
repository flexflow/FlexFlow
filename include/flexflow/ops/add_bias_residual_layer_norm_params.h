#pragma once

#include "flexflow/ffconst.h"
#include "flexflow/fftype.h"
#include "flexflow/parallel_tensor.h"

namespace FlexFlow {

struct AddBiasResidualLayerNormParams {
  LayerID layer_guid;
  std::vector<int> axes;
  bool elementwise_affine;
  float eps;
  bool use_bias;
  char name[MAX_OPNAME];
  bool is_valid(
      std::pair<ParallelTensorShape, ParallelTensorShape> const &) const;
};

bool operator==(AddBiasResidualLayerNormParams const &,
                AddBiasResidualLayerNormParams const &);

} // namespace FlexFlow

namespace std {
template <>
struct hash<FlexFlow::AddBiasResidualLayerNormParams> {
  size_t operator()(FlexFlow::AddBiasResidualLayerNormParams const &) const;
};
} // namespace std
