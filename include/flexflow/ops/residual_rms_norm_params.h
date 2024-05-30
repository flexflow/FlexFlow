#ifndef _FLEXFLOW_RESIDUAL_RMSNORM_PARAMS_H
#define _FLEXFLOW_RESIDUAL_RMSNORM_PARAMS_H

#include "flexflow/ffconst.h"
#include "flexflow/fftype.h"
#include "flexflow/parallel_tensor.h"

namespace FlexFlow {

struct ResidualRMSNormParams {
  LayerID layer_guid;
  float eps;
  int dim;
  char name[MAX_OPNAME];
  bool is_valid(
      std::pair<ParallelTensorShape, ParallelTensorShape> const &input) const;
};

bool operator==(ResidualRMSNormParams const &, ResidualRMSNormParams const &);

} // namespace FlexFlow

namespace std {
template <>
struct hash<FlexFlow::ResidualRMSNormParams> {
  size_t operator()(FlexFlow::ResidualRMSNormParams const &) const;
};
} // namespace std

#endif // _FLEXFLOW_RESIDUAL_RMSNORM_PARAMS_H