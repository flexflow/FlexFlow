#ifndef _FLEXFLOW_PCG_INCLUDE_PCG_FILE_FORMAT_V1_OPS_BROADCAST_H
#define _FLEXFLOW_PCG_INCLUDE_PCG_FILE_FORMAT_V1_OPS_BROADCAST_H

#include "utils/stack_vector.h"
#include "utils/visitable.h"

namespace FlexFlow {

struct V1BroadcastAttrs {
  req<stack_vector<int, MAX_TENSOR_DIM>> target_dims;
};
FF_VISITABLE_STRUCT(V1BroadcastAttrs, target_dims);
CHECK_IS_JSONABLE(V1BroadcastAttrs);

V1BroadcastAttrs to_v1(BroadcastAttrs const &attrs);

} // namespace FlexFlow

#endif
