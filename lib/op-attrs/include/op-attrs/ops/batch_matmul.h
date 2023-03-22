#ifndef _FF_OP_META_BATCH_MATMUL_ATTRS_H
#define _FF_OP_META_BATCH_MATMUL_ATTRS_H

#include "op-attrs/parallel_tensor_shape.h"
#include "op-attrs/ops/binary_op.h"
#include "visit_struct/visit_struct.hpp"

namespace FlexFlow {

struct BatchMatmulAttrs : public BinaryOpAttrs {
public:
  bool is_valid(ParallelTensorShape const &rhs_input_shape, ParallelTensorShape const &lhs_input_shape) const override;
  ParallelTensorShape output_shape(ParallelTensorShape const &rhs_input_shape, ParallelTensorShape const &lhs_input_shape) const override;
  OperatorType op_type() const override;
public:
  int a_seq_length_dim, b_seq_length_dim;
};

bool operator==(BatchMatmulAttrs const &, BatchMatmulAttrs const &);
bool operator<(BatchMatmulAttrs const &, BatchMatmulAttrs const &);

}

VISITABLE_STRUCT(::FlexFlow::BatchMatmulAttrs, a_seq_length_dim, b_seq_length_dim);

namespace std {
template <>
struct hash<::FlexFlow::BatchMatmulAttrs> {
  size_t operator()(::FlexFlow::BatchMatmulAttrs const &) const;
};
} 

#endif 
