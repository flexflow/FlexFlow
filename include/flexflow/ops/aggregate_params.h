#ifndef _FLEXFLOW_AGGREGATE_PARAMS_H
#define _FLEXFLOW_AGGREGATE_PARAMS_H

#include "flexflow/ffconst.h"
#include "flexflow/parallel_tensor.h"

namespace FlexFlow {

struct AggregateParams {
  int n;
  float lambda_bal;
  char name[MAX_OPNAME];
  bool is_valid(std::vector<ParallelTensorShape> const &) const;
};
bool operator==(AggregateParams const &, AggregateParams const &);

} // namespace FlexFlow

namespace std {
template <>
struct hash<FlexFlow::AggregateParams> {
  size_t operator()(FlexFlow::AggregateParams const &) const;
};
} // namespace std

#endif // _FLEXFLOW_AGGREGATE_PARAMS_H
