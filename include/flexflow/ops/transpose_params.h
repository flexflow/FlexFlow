#pragma once

#include "flexflow/parallel_tensor.h"

namespace FlexFlow {

struct TransposeParams {
  std::vector<int> perm;
  char name[MAX_OPNAME];
  bool is_valid(ParallelTensorShape const &) const;
};

bool operator==(TransposeParams const &, TransposeParams const &);

} // namespace FlexFlow

namespace std {
template <>
struct hash<FlexFlow::TransposeParams> {
  size_t operator()(FlexFlow::TransposeParams const &) const;
};
} // namespace std
