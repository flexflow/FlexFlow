#ifndef _FLEXFLOW_DROPOUT_PARAMS_H
#define _FLEXFLOW_DROPOUT_PARAMS_H

#include "flexflow/ffconst.h"
#include "flexflow/parallel_tensor.h"

namespace FlexFlow {

struct DropoutParams {
  float rate;
  unsigned long long seed;
  char name[MAX_OPNAME];
  bool is_valid(ParallelTensorShape const &) const;
};
bool operator==(DropoutParams const &, DropoutParams const &);

} // namespace FlexFlow

namespace std {
template <>
struct hash<FlexFlow::DropoutParams> {
  size_t operator()(FlexFlow::DropoutParams const &) const;
};
} // namespace std

#endif // _FLEXFLOW_DROPOUT_PARAMS_H
