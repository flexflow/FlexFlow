#pragma once

#include "flexflow/parallel_tensor.h"

namespace FlexFlow {

struct BatchMatmulParams {
  int a_seq_length_dim, b_seq_length_dim;
  char name[MAX_OPNAME];
  bool is_valid(
      std::pair<ParallelTensorShape, ParallelTensorShape> const &) const;
};

bool operator==(BatchMatmulParams const &, BatchMatmulParams const &);

} // namespace FlexFlow

namespace std {
template <>
struct hash<FlexFlow::BatchMatmulParams> {
  size_t operator()(FlexFlow::BatchMatmulParams const &) const;
};
} // namespace std
