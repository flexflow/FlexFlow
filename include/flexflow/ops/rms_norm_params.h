#ifndef _FLEXFLOW_RMSNORM_PARAMS_H
#define _FLEXFLOW_RMSNORM_PARAMS_H

#include "flexflow/ffconst.h"
#include "flexflow/fftype.h"
#include "flexflow/parallel_tensor.h"

namespace FlexFlow {

struct RMSNormParams {
  LayerID layer_guid;
  float eps;
  int dim;
  char name[MAX_OPNAME];
  bool is_valid(ParallelTensorShape const &) const;
};

bool operator==(RMSNormParams const &, RMSNormParams const &);

} // namespace FlexFlow

namespace std {
template <>
struct hash<FlexFlow::RMSNormParams> {
  size_t operator()(FlexFlow::RMSNormParams const &) const;
};
} // namespace std

#endif // _FLEXFLOW_RMSNORM_PARAMS_H