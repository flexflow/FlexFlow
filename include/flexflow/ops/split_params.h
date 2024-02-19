#ifndef _FLEXFLOW_SPLIT_PARAMS_H
#define _FLEXFLOW_SPLIT_PARAMS_H

#include "flexflow/parallel_tensor.h"

namespace FlexFlow {

struct SplitParams {
  std::vector<int> splits;
  int legion_axis;
  char name[MAX_OPNAME];
  bool is_valid(ParallelTensorShape const &) const;
};

bool operator==(SplitParams const &, SplitParams const &);

} // namespace FlexFlow

namespace std {
template <>
struct hash<FlexFlow::SplitParams> {
  size_t operator()(FlexFlow::SplitParams const &) const;
};
} // namespace std

#endif // _FLEXFLOW_SPLIT_PARAMS_H
