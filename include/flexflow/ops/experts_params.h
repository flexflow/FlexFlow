#pragma once

#include "flexflow/parallel_tensor.h"

namespace FlexFlow {

struct ExpertsParams {
  bool is_valid(
      std::pair<ParallelTensorShape, ParallelTensorShape> const &) const;
  int num_experts;
  int experts_start_idx;
  int experts_num_layers;
  int experts_output_dim_size;
  int experts_internal_dim_size;
  bool use_bias;
  ActiMode activation; 
};

bool operator==(ExpertsParams const &, ExpertsParams const &);

} // namespace FlexFlow

namespace std {
template <>
struct hash<FlexFlow::ExpertsParams> {
  size_t operator()(FlexFlow::ExpertsParams const &) const;
};
} // namespace std
