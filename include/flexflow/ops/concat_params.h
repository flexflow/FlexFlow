#ifndef _FLEXFLOW_CONCAT_PARAMS_H
#define _FLEXFLOW_CONCAT_PARAMS_H

#include "flexflow/parallel_tensor.h"

namespace FlexFlow {

struct ConcatParams {
  int axis;
  char name[MAX_OPNAME];
  bool is_valid(std::vector<ParallelTensorShape> const &) const;
};

bool operator==(ConcatParams const &, ConcatParams const &);

} // namespace FlexFlow

namespace std {
template <>
struct hash<FlexFlow::ConcatParams> {
  size_t operator()(FlexFlow::ConcatParams const &) const;
};
} // namespace std

#endif // _FLEXFLOW_CONCAT_PARAMS_H
