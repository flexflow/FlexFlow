#ifndef _FLEXFLOW_FUSED_PARALLEL_OP_PARAMS_H
#define _FLEXFLOW_FUSED_PARALLEL_OP_PARAMS_H


#include "op-meta/parallel_op_info.h"
#include <vector>
#include "op-meta/parallel_tensor_shape.h"

namespace FlexFlow {

struct FusedParallelOpParams {
public:
  bool is_valid(ParallelTensorShape const &) const;

  using AsConstTuple = std::tuple<std::vector<ParallelOpInfo>>;
  AsConstTuple as_tuple() const;
public:
  std::vector<ParallelOpInfo> parallel_ops;
};
bool operator==(FusedParallelOpParams const &, FusedParallelOpParams const &);
bool operator<(FusedParallelOpParams const &, FusedParallelOpParams const &);

} 

namespace std {
template <>
struct hash<FlexFlow::FusedParallelOpParams> {
  size_t operator()(FlexFlow::FusedParallelOpParams const &) const;
};
} 

#endif 
