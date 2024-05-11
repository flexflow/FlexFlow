#ifndef _FLEXFLOW_GUMBEL_TOPK_PARAMS_H
#define _FLEXFLOW_GUMBEL_TOPK_PARAMS_H

#include "flexflow/ffconst.h"
#include "flexflow/fftype.h"
#include "flexflow/parallel_tensor.h"

namespace FlexFlow {

struct GumbelTopKParams {
  LayerID layer_guid;
  int k;
  bool sorted;
  bool speculative_decoding;
  char name[MAX_OPNAME];
  bool is_valid(ParallelTensorShape const &) const;
};
bool operator==(GumbelTopKParams const &, GumbelTopKParams const &);

} // namespace FlexFlow

namespace std {
template <>
struct hash<FlexFlow::GumbelTopKParams> {
  size_t operator()(FlexFlow::GumbelTopKParams const &) const;
};
} // namespace std

#endif // _FLEXFLOW_GUMBEL_TOPK_PARAMS_H
