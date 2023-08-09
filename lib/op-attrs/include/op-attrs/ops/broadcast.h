#ifndef _FLEXFLOW_INCLUDE_OPATTRS_OPS_BROADCAST_H
#define _FLEXFLOW_INCLUDE_OPATTRS_OPS_BROADCAST_H

#include "core.h"
#include "utils/stack_vector.h"
#include "utils/visitable.h"

namespace FlexFlow {

struct BroadcastAttrs {
  req<stack_vector<int, MAX_TENSOR_DIM>> target_dims;
};
FF_VISITABLE_STRUCT(BroadcastAttrs, target_dims);

CHECK_VALID_OP_ATTR(BroadcastAttrs);

} // namespace FlexFlow

#endif
