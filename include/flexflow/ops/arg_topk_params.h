#ifndef _FLEXFLOW_ARG_TOPK_PARAMS_H
#define _FLEXFLOW_ARG_TOPK_PARAMS_H

#include "flexflow/ffconst.h"
#include "flexflow/fftype.h"
#include "flexflow/parallel_tensor.h"

namespace FlexFlow {

struct ArgTopKParams {
  LayerID layer_guid;
  int k;
  bool sorted;
  bool speculative_decoding;
  char name[MAX_OPNAME];
  bool is_valid(ParallelTensorShape const &) const;
};
bool operator==(ArgTopKParams const &, ArgTopKParams const &);

} // namespace FlexFlow

namespace std {
template <>
struct hash<FlexFlow::ArgTopKParams> {
  size_t operator()(FlexFlow::ArgTopKParams const &) const;
};
} // namespace std

#endif // _FLEXFLOW_ARG_TOPK_PARAMS_H
