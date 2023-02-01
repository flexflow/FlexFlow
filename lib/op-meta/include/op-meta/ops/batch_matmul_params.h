#ifndef _FF_OP_META_BATCH_MATMUL_PARAMS_H
#define _FF_OP_META_BATCH_MATMUL_PARAMS_H

#include "op-meta/parallel_tensor_shape.h"
#include "op-meta/ops/binary_op.h"

namespace FlexFlow {

struct BatchMatmulParams : public BinaryOpParams {
public:
  using AsConstTuple = std::tuple<int, int>;
  AsConstTuple as_tuple() const;

  bool is_valid(ParallelTensorShape const &, ParallelTensorShape const &) const override;
  OperatorType op_type() const override;
public:
  int a_seq_length_dim, b_seq_length_dim;
};

bool operator==(BatchMatmulParams const &, BatchMatmulParams const &);
bool operator<(BatchMatmulParams const &, BatchMatmulParams const &);

} 

namespace std {
template <>
struct hash<FlexFlow::BatchMatmulParams> {
  size_t operator()(FlexFlow::BatchMatmulParams const &) const;
};
} 

#endif // _FF_OP_META_BATCH_MATMUL_PARAMS_H
