#ifndef _FLEXFLOW_GROUPBY_PARAMS_H
#define _FLEXFLOW_GROUPBY_PARAMS_H

#include "flexflow/ffconst.h"
#include "flexflow/parallel_tensor.h"

namespace FlexFlow {

struct Group_byParams {
  int n;
  float alpha;
  char name[MAX_OPNAME];
  bool is_valid(
      std::pair<ParallelTensorShape, ParallelTensorShape> const &) const;
};
bool operator==(Group_byParams const &, Group_byParams const &);

} // namespace FlexFlow

namespace std {
template <>
struct hash<FlexFlow::Group_byParams> {
  size_t operator()(FlexFlow::Group_byParams const &) const;
};
} // namespace std

#endif // _FLEXFLOW_GROUPBY_PARAMS_H
