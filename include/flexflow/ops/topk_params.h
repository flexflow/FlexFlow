#ifndef _FLEXFLOW_TOPK_PARAMS_H
#define _FLEXFLOW_TOPK_PARAMS_H

#include "flexflow/ffconst.h"
#include "flexflow/parallel_tensor.h"

namespace FlexFlow {

struct TopKParams {
  int k;
  bool sorted;
  char name[MAX_OPNAME];
  bool is_valid(ParallelTensorShape const &) const;
};
bool operator==(TopKParams const &, TopKParams const &);

} // namespace FlexFlow

namespace std {
template <>
struct hash<FlexFlow::TopKParams> {
  size_t operator()(FlexFlow::TopKParams const &) const;
};
} // namespace std

#endif // _FLEXFLOW_TOPK_PARAMS_H
