#ifndef _FLEXFLOW_BEAM_TOPK_PARAMS_H
#define _FLEXFLOW_BEAM_TOPK_PARAMS_H

#include "flexflow/ffconst.h"
#include "flexflow/fftype.h"
#include "flexflow/parallel_tensor.h"

namespace FlexFlow {

struct BeamTopKParams {
  LayerID layer_guid;
  bool sorted;
  int max_beam_width;
  char name[MAX_OPNAME];
  bool is_valid(ParallelTensorShape const &) const;
};
bool operator==(BeamTopKParams const &, BeamTopKParams const &);

} // namespace FlexFlow

namespace std {
template <>
struct hash<FlexFlow::BeamTopKParams> {
  size_t operator()(FlexFlow::BeamTopKParams const &) const;
};
} // namespace std

#endif // _FLEXFLOW_BEAM_TOPK_PARAMS_H
