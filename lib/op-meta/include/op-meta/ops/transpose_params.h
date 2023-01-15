#ifndef _FLEXFLOW_OP_META_OPS_TRANSPOSE_PARAMS_H
#define _FLEXFLOW_OP_META_OPS_TRANSPOSE_PARAMS_H

#include "op-meta/parallel_tensor_shape.h"

namespace FlexFlow {

struct TransposeParams {
  std::vector<int> perm;
  bool is_valid(ParallelTensorShape const &) const;
};

bool operator==(TransposeParams const &, TransposeParams const &);

} // namespace FlexFlow

namespace std {
template <>
struct hash<FlexFlow::TransposeParams> {
  size_t operator()(FlexFlow::TransposeParams const &) const;
};
} 

#endif // _FLEXFLOW_OP_META_OPS_TRANSPOSE_PARAMS_H
