#ifndef _FLEXFLOW_PCG_INCLUDE_PCG_FILE_FORMAT_V1_OPS_REDUCE_ATTRS_H
#define _FLEXFLOW_PCG_INCLUDE_PCG_FILE_FORMAT_V1_OPS_REDUCE_ATTRS_H

#include "op-attrs/ff_dim.h"
#include "op-attrs/op.h"
#include "op-attrs/ops/reduce.h"
#include "utils/visitable.h"

namespace FlexFlow {

struct V1ReduceAttrs {
  // The size of this vector is <= MAX_TENSOR_DIMS.
  req<std::vector<ff_dim_t>> axes;
  req<Op> op_type;
  req<bool> keepdims;
};
FF_VISITABLE_STRUCT(V1ReduceAttrs, axes, op_type, keepdims);
CHECK_IS_JSONABLE(V1ReduceAttrs);

V1ReduceAttrs to_v1(ReduceAttrs const &attrs);

} // namespace FlexFlow

#endif
