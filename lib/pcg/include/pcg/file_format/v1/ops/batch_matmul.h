#ifndef _FLEXFLOW_PCG_INCLUDE_PCG_FILE_FORMAT_V1_OPS_BATCH_MATMUL_ATTRS_H
#define _FLEXFLOW_PCG_INCLUDE_PCG_FILE_FORMAT_V1_OPS_BATCH_MATMUL_ATTRS_H

#include "op-attrs/ops/batch_matmul.h"
#include "utils/visitable.h"

namespace FlexFlow {

struct V1BatchMatmulAttrs {
  req<int> a_seq_length_dim, b_seq_length_dim;
};
FF_VISITABLE_STRUCT(V1BatchMatmulAttrs, a_seq_length_dim, b_seq_length_dim);
CHECK_IS_JSONABLE(V1BatchMatmulAttrs);

V1BatchMatmulAttrs to_v1(BatchMatmulAttrs const &attrs);

} // namespace FlexFlow

#endif
