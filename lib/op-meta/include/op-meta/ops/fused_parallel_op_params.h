#ifndef _FLEXFLOW_FUSED_PARALLEL_OP_PARAMS_H
#define _FLEXFLOW_FUSED_PARALLEL_OP_PARAMS_H

#include "op-meta/parallel_op_info.h"
#include <vector>
#include "op-meta/parallel_tensor_shape.h"
#include "op-meta/ops/unary_op.h"
#include "visit_struct/visit_struct.hpp"

namespace FlexFlow {
namespace opmeta {

struct FusedParallelOpParams : public UnaryOpParams {
public:
  ParallelTensorShape output_shape(ParallelTensorShape const &input_shape) const override;
  OperatorType op_type() const override;

public:
  std::vector<ParallelOpInfo> parallel_ops;
};
bool operator==(FusedParallelOpParams const &, FusedParallelOpParams const &);
bool operator<(FusedParallelOpParams const &, FusedParallelOpParams const &);

} 
}

VISITABLE_STRUCT(::FlexFlow::opmeta::FusedParallelOpParams, parallel_ops);

namespace std {
template <>
struct hash<::FlexFlow::opmeta::FusedParallelOpParams> {
  size_t operator()(::FlexFlow::opmeta::FusedParallelOpParams const &) const;
};
} 

#endif 
