#ifndef _FLEXFLOW_OP_META_OPS_TRANSPOSE_PARAMS_H
#define _FLEXFLOW_OP_META_OPS_TRANSPOSE_PARAMS_H

#include "op-meta/parallel_tensor_shape.h"

namespace FlexFlow {

struct TransposeParams {
public:
  bool is_valid(ParallelTensorShape const &) const;

  using AsConstTuple = std::tuple<std::vector<int>>;
  AsConstTuple as_tuple() const;
public:
  std::vector<int> perm;
};

bool operator==(TransposeParams const &, TransposeParams const &);
bool operator<(TransposeParams const &, TransposeParams const &);

} 

namespace std {
template <>
struct hash<FlexFlow::TransposeParams> {
  size_t operator()(FlexFlow::TransposeParams const &) const;
};
} 

#endif 
