#ifndef _FLEXFLOW_LIB_OP_ATTRS_INCLUDE_OP_ATTRS_OPS_BATCH_MATMUL_H
#define _FLEXFLOW_LIB_OP_ATTRS_INCLUDE_OP_ATTRS_OPS_BATCH_MATMUL_H

#include "op-attrs/ops/batch_matmul.dtg.h"
#include "op-attrs/parallel_tensor_shape.dtg.h"

namespace FlexFlow {

bool is_valid(BatchMatmulAttrs const &,
              ParallelTensorShape const &,
              ParallelTensorShape const &);

} // namespace FlexFlow

#endif
