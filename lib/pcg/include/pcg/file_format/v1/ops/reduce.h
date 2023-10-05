#ifndef _FLEXFLOW_PCG_INCLUDE_PCG_FILE_FORMAT_V1_OPS_REDUCE_ATTRS_H
#define _FLEXFLOW_PCG_INCLUDE_PCG_FILE_FORMAT_V1_OPS_REDUCE_ATTRS_H

#include "op-attrs/ops/reduce.h"
#include "pcg/file_format/v1/ff_dim.h"
#include "pcg/file_format/v1/op.h"
#include "utils/visitable.h"

namespace FlexFlow {

struct V1ReduceAttrs {
  // The size of this vector is <= MAX_TENSOR_DIMS.
  req<std::vector<int>> axes;
  req<V1Op> op_type;
  req<bool> keepdims;
};
FF_VISITABLE_STRUCT(V1ReduceAttrs, axes, op_type, keepdims);
CHECK_IS_JSONABLE(V1ReduceAttrs);

V1ReduceAttrs to_v1(ReduceAttrs const &a);
ReduceAttrs from_v1(V1ReduceAttrs const &va);

} // namespace FlexFlow

#endif
