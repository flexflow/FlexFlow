#ifndef _FLEXFLOW_RESHAPE_PARAMS_H
#define _FLEXFLOW_RESHAPE_PARAMS_H

#include "flexflow/ffconst.h"
#include "flexflow/fftype.h"
#include "flexflow/parallel_tensor.h"

namespace FlexFlow {

struct ReshapeParams {
  std::vector<int> shape;
  LayerID layer_guid;
  char name[MAX_OPNAME];

  bool is_valid(ParallelTensorShape const &) const;
};
bool operator==(ReshapeParams const &, ReshapeParams const &);

} // namespace FlexFlow

namespace std {
template <>
struct hash<FlexFlow::ReshapeParams> {
  size_t operator()(FlexFlow::ReshapeParams const &) const;
};
} // namespace std

#endif // _FLEXFLOW_RESHAPE_PARAMS_H
