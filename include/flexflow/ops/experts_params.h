#pragma once

#include "flexflow/ffconst.h"
#include "flexflow/fftype.h"
#include "flexflow/operator.h"
#include "flexflow/parallel_tensor.h"

namespace FlexFlow {

struct ExpertsParams {
  LayerID layer_guid;
  int num_experts;
  int experts_start_idx;
  int experts_output_dim_size;
  float alpha;
  int experts_num_layers;
  int experts_internal_dim_size;
  bool use_bias;
  ActiMode activation;
  char name[MAX_OPNAME];

  bool is_valid(std::vector<ParallelTensorShape> const &) const;
};

bool operator==(ExpertsParams const &, ExpertsParams const &);

} // namespace FlexFlow

namespace std {
template <>
struct hash<FlexFlow::ExpertsParams> {
  size_t operator()(FlexFlow::ExpertsParams const &) const;
};
} // namespace std
