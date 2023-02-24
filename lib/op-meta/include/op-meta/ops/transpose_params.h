#ifndef _FLEXFLOW_OP_META_OPS_TRANSPOSE_PARAMS_H
#define _FLEXFLOW_OP_META_OPS_TRANSPOSE_PARAMS_H

#include "op-meta/parallel_tensor_shape.h"
#include "op-meta/ops/unary_op.h"
#include "visit_struct/visit_struct.hpp"

namespace FlexFlow {
namespace opmeta {

struct TransposeParams : public UnaryOpParams {
public:
  ParallelTensorShape output_shape(ParallelTensorShape const &input_shape) const override;
  OperatorType op_type() const override;
public:
  std::vector<int> perm;
};

bool operator==(TransposeParams const &, TransposeParams const &);
bool operator<(TransposeParams const &, TransposeParams const &);

} 
}

VISITABLE_STRUCT(::FlexFlow::opmeta::TransposeParams, perm);

namespace std {
template <>
struct hash<::FlexFlow::opmeta::TransposeParams> {
  size_t operator()(::FlexFlow::opmeta::TransposeParams const &) const;
};
} 

#endif 
