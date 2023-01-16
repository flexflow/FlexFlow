#pragma once

#include "flexflow/parallel_tensor.h"

namespace FlexFlow {

struct GatherParams {
  int legion_dim;
  bool is_valid(
      std::pair<ParallelTensorShape, ParallelTensorShape> const &input) const;
};

bool operator==(GatherParams const &, GatherParams const &);

} // namespace FlexFlow

namespace std {
template <>
struct hash<FlexFlow::GatherParams> {
  size_t operator()(FlexFlow::GatherParams const &) const;
};
} // namespace std
