#ifndef _FLEXFLOW_STOP_GRAD_PARAMS_H
#define _FLEXFLOW_STOP_GRAD_PARAMS_H

#include "flexflow/ffconst.h"
#include "flexflow/parallel_tensor.h"

namespace FlexFlow {

struct StopGradParams {
  bool is_valid(ParallelTensorShape const &) const;
};

bool operator==(StopGradParams const &, StopGradParams const &);

} // namespace FlexFlow

namespace std {
template <>
struct hash<FlexFlow::StopGradParams> {
  size_t operator()(FlexFlow::StopGradParams const &) const;
};
} // namespace std

#endif // _FLEXFLOW_STOP_GRAD_PARAMS_H
