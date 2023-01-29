#ifndef _FLEXFLOW_REDUCTION_PARAMS_H
#define _FLEXFLOW_REDUCTION_PARAMS_H

#include "op-meta/parallel_tensor_shape.h"

namespace FlexFlow {

struct ReductionParams {
public:
  bool is_valid(ParallelTensorShape const &) const;

  using AsConstTuple = std::tuple<int, int>;
  AsConstTuple as_tuple() const;
public:
  int reduction_legion_dim;
  int reduction_degree;
};
bool operator==(ReductionParams const &, ReductionParams const &);
bool operator<(ReductionParams const &, ReductionParams const &);

} 

namespace std {
template <>
struct hash<FlexFlow::ReductionParams> {
  size_t operator()(FlexFlow::ReductionParams const &) const;
};
} // namespace std

#endif // _FLEXFLOW_REDUCTION_PARAMS_H
