#ifndef _FLEXFLOW_PCG_INCLUDE_PCG_FILE_FORMAT_V1_OPS_BROADCAST_H
#define _FLEXFLOW_PCG_INCLUDE_PCG_FILE_FORMAT_V1_OPS_BROADCAST_H

#include "op-attrs/ops/broadcast.h"
#include "utils/visitable.h"

namespace FlexFlow {

struct V1BroadcastAttrs {
  // The size of this vector must be <= MAX_TENSOR_DIM
  req<std::vector<int>> target_dims;
};
FF_VISITABLE_STRUCT(V1BroadcastAttrs, target_dims);
CHECK_IS_JSONABLE(V1BroadcastAttrs);

V1BroadcastAttrs to_v1(BroadcastAttrs const &attrs);

} // namespace FlexFlow

#endif
