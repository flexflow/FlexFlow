#ifndef _FLEXFLOW_COMBINE_PARAMS_H
#define _FLEXFLOW_COMBINE_PARAMS_H

#include "op-meta/parallel_tensor_shape.h"

namespace FlexFlow {

struct CombineParams {
  int combine_legion_dim;
  int combine_degree;
  bool is_valid(ParallelTensorShape const &) const;

  using AsConstTuple = std::tuple<int, int>;
  AsConstTuple as_tuple() const;
};
bool operator==(CombineParams const &, CombineParams const &);
bool operator<(CombineParams const &, CombineParams const &);

} // namespace FlexFlow

namespace std {
template <>
struct hash<FlexFlow::CombineParams> {
  size_t operator()(FlexFlow::CombineParams const &) const;
};
} // namespace std

#endif // _FLEXFLOW_COMBINE_PARAMS_H
