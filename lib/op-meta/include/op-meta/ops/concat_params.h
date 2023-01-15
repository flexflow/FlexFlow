#ifndef _FLEXFLOW_CONCAT_PARAMS_H
#define _FLEXFLOW_CONCAT_PARAMS_H

#include "op-meta/parallel_tensor_shape.h"

namespace FlexFlow {

struct ConcatParams {
  int axis;

  bool is_valid(std::vector<ParallelTensorShape> const &) const;

  using AsConstTuple = std::tuple<int>;
  AsConstTuple as_tuple() const;
};

bool operator==(ConcatParams const &, ConcatParams const &);
bool operator<(ConcatParams const &, ConcatParams const &);

} // namespace FlexFlow

namespace std {
template <>
struct hash<FlexFlow::ConcatParams> {
  size_t operator()(FlexFlow::ConcatParams const &) const;
};
} // namespace std

#endif // _FLEXFLOW_CONCAT_PARAMS_H
