#ifndef _FF_OP_META_BATCH_MATMUL_PARAMS_H
#define _FF_OP_META_BATCH_MATMUL_PARAMS_H

#include "op-meta/parallel_tensor_shape.h"
#include "op-meta/ops/binary_op.h"
#include "visit_struct/visit_struct.hpp"

namespace FlexFlow {
namespace opmeta {

struct BatchMatmulParams : public BinaryOpParams {
public:
  bool is_valid(ParallelTensorShape const &rhs_input_shape, ParallelTensorShape const &lhs_input_shape) const override;
  ParallelTensorShape output_shape(ParallelTensorShape const &rhs_input_shape, ParallelTensorShape const &lhs_input_shape) const override;
  OperatorType op_type() const override;
public:
  int a_seq_length_dim, b_seq_length_dim;
};

bool operator==(BatchMatmulParams const &, BatchMatmulParams const &);
bool operator<(BatchMatmulParams const &, BatchMatmulParams const &);

} 
}

VISITABLE_STRUCT(::FlexFlow::opmeta::BatchMatmulParams, a_seq_length_dim, b_seq_length_dim);

namespace std {
template <>
struct hash<::FlexFlow::opmeta::BatchMatmulParams> {
  size_t operator()(::FlexFlow::opmeta::BatchMatmulParams const &) const;
};
} 

#endif // _FF_OP_META_BATCH_MATMUL_PARAMS_H
