#ifndef _FLEXFLOW_SOFTMAX_PARAMS_H
#define _FLEXFLOW_SOFTMAX_PARAMS_H

#include "flexflow/parallel_tensor.h"

namespace FlexFlow {

struct SoftmaxParams {
  LayerID layer_guid;
  int dim;
  char name[MAX_OPNAME];
  bool is_valid(ParallelTensorShape const &) const;
};
bool operator==(SoftmaxParams const &, SoftmaxParams const &);

} // namespace FlexFlow

namespace std {
template <>
struct hash<FlexFlow::SoftmaxParams> {
  size_t operator()(FlexFlow::SoftmaxParams const &) const;
};
} // namespace std

#endif // _FLEXFLOW_SOFTMAX_PARAMS_H
