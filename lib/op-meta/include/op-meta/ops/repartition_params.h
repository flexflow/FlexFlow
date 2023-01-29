#ifndef _FLEXFLOW_PARTITION_PARAMS_H
#define _FLEXFLOW_PARTITION_PARAMS_H

#include "op-meta/parallel_tensor_shape.h"

namespace FlexFlow {

struct RepartitionParams {
public:
  bool is_valid(ParallelTensorShape const &) const;

  using AsConstTuple = std::tuple<int, int>;
  AsConstTuple as_tuple() const;

public:
  int repartition_legion_dim;
  int repartition_degree;
};
bool operator==(RepartitionParams const &, RepartitionParams const &);
bool operator<(RepartitionParams const &, RepartitionParams const &);

} // namespace FlexFlow

namespace std {
template <>
struct hash<FlexFlow::RepartitionParams> {
  size_t operator()(FlexFlow::RepartitionParams const &) const;
};
} // namespace std

#endif // _FLEXFLOW_PARTITION_PARAMS_H
