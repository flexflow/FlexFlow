#ifndef _FLEXFLOW_FUSED_PARALLEL_OP_PARAMS_H
#define _FLEXFLOW_FUSED_PARALLEL_OP_PARAMS_H

#include "op-meta/parallel_op_info.h"
#include <vector>
#include "op-meta/parallel_tensor_shape.h"
#include "op-meta/ops/op_params.h"

namespace FlexFlow {

struct FusedParallelOpParams : public OpParamsInterface {
public:
  using AsConstTuple = std::tuple<std::vector<ParallelOpInfo>>;
  AsConstTuple as_tuple() const;

  int num_outputs(std::vector<ParallelTensorShape> const &inputs) const override;
  bool is_valid(std::vector<ParallelTensorShape> const &) const override;
  OperatorType op_type() const override;

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
