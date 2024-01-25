#ifndef _FLEXFLOW_AGGREGATE_SPEC_PARAMS_H
#define _FLEXFLOW_AGGREGATE_SPEC_PARAMS_H

#include "flexflow/ffconst.h"
#include "flexflow/parallel_tensor.h"

namespace FlexFlow {

struct AggregateSpecParams {
  int n;
  float lambda_bal;
  char name[MAX_OPNAME];
  bool is_valid(ParallelTensorShape const &) const;
};
bool operator==(AggregateSpecParams const &, AggregateSpecParams const &);

} // namespace FlexFlow

namespace std {
template <>
struct hash<FlexFlow::AggregateSpecParams> {
  size_t operator()(FlexFlow::AggregateSpecParams const &) const;
};
} // namespace std

#endif // _FLEXFLOW_AGGREGATE_SPEC_PARAMS_H
