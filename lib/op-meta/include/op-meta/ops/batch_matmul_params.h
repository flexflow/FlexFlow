#ifndef _FF_OP_META_BATCH_MATMUL_PARAMS_H
#define _FF_OP_META_BATCH_MATMUL_PARAMS_H

#include "op-meta/parallel_tensor_shape.h"

namespace FlexFlow {

struct BatchMatmulParams {
  int a_seq_length_dim, b_seq_length_dim;
  bool is_valid(
      std::pair<ParallelTensorShape, ParallelTensorShape> const &) const;

  using AsConstTuple = std::tuple<int, int>;
  AsConstTuple as_tuple() const;
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
