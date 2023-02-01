#ifndef _FLEXFLOW_OP_META_OPS_TRANSPOSE_PARAMS_H
#define _FLEXFLOW_OP_META_OPS_TRANSPOSE_PARAMS_H

#include "op-meta/parallel_tensor_shape.h"
#include "op-meta/ops/unary_op.h"

namespace FlexFlow {

struct TransposeParams : public UnaryOpParams {
public:
  using AsConstTuple = std::tuple<std::vector<int>>;
  AsConstTuple as_tuple() const;

  bool is_valid(ParallelTensorShape const &) const;
  OperatorType op_type() const;
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
