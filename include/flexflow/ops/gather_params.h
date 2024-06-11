#ifndef _FLEXFLOW_GATHER_PARAMS_H
#define _FLEXFLOW_GATHER_PARAMS_H

#include "flexflow/ffconst.h"
#include "flexflow/fftype.h"
#include "flexflow/parallel_tensor.h"

namespace FlexFlow {

struct GatherParams {
  int legion_dim;
  LayerID layer_guid;
  char name[MAX_OPNAME];
  bool is_valid(
      std::pair<ParallelTensorShape, ParallelTensorShape> const &input) const;
};

bool operator==(GatherParams const &, GatherParams const &);

} // namespace FlexFlow

namespace std {
template <>
struct hash<FlexFlow::GatherParams> {
  size_t operator()(FlexFlow::GatherParams const &) const;
};
} // namespace std

#endif // _FLEXFLOW_GATHER_PARAMS_H
