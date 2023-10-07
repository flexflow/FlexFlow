#ifndef _FF_OP_META_BATCH_MATMUL_ATTRS_H
#define _FF_OP_META_BATCH_MATMUL_ATTRS_H

#include "core.h"
#include "op-attrs/parallel_tensor_shape.h"
#include "utils/visitable.h"

namespace FlexFlow {

struct BatchMatmulAttrs {
  req<int> a_seq_length_dim, b_seq_length_dim;
  bool is_valid(ParallelTensorShape const &,
                                     ParallelTensorShape const &);
};
FF_VISITABLE_STRUCT(BatchMatmulAttrs, a_seq_length_dim, b_seq_length_dim);

CHECK_VALID_OP_ATTR(BatchMatmulAttrs);

ParallelTensorShape get_output_shape(BatchMatmulAttrs const &,
                                     ParallelTensorShape const &,
                                     ParallelTensorShape const &);

} // namespace FlexFlow

#endif
