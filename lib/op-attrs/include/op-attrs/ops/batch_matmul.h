#ifndef _FF_OP_META_BATCH_MATMUL_ATTRS_H
#define _FF_OP_META_BATCH_MATMUL_ATTRS_H

#include "core.h"
#include "op-attrs/parallel_tensor_shape.h"
#include "utils/visitable.h"

namespace FlexFlow {

struct BatchMatmulAttrs {
  req<int> a_seq_length_dim, b_seq_length_dim;
};
FF_VISITABLE_STRUCT(BatchMatmulAttrs, a_seq_length_dim, b_seq_length_dim);

CHECK_VALID_OP_ATTR(BatchMatmulAttrs);

} // namespace FlexFlow

#endif
