#ifndef _FLEXFLOW_FUSED_PARALLEL_OP_PARAMS_H
#define _FLEXFLOW_FUSED_PARALLEL_OP_PARAMS_H

#include "flexflow/parallel_ops/parallel_op_info.h"

namespace FlexFlow {

struct FusedParallelOpParams {
  std::vector<ParallelOpInfo> parallel_ops;
  bool is_valid(std::vector<ParallelTensorShape> const &) const;
};
bool operator==(FusedParallelOpParams const &, FusedParallelOpParams const &);

} // namespace FlexFlow

namespace std {
template <>
struct hash<FlexFlow::FusedParallelOpParams> {
  size_t operator()(FlexFlow::FusedParallelOpParams const &) const;
};
} // namespace std

#endif // _FLEXFLOW_FUSED_PARALLEL_OP_PARAMS_H
