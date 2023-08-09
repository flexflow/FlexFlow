#ifndef _FLEXFLOW_SPLIT_ATTRS_H
#define _FLEXFLOW_SPLIT_ATTRS_H

#include "core.h"
#include "op-attrs/ff_dim.h"
#include "op-attrs/parallel_tensor_shape.h"
#include "utils/visitable.h"

namespace FlexFlow {

struct SplitAttrs {
  req<stack_vector<int, MAX_NUM_OUTPUTS>> splits;
  ff_dim_t axis;
};
FF_VISITABLE_STRUCT(SplitAttrs, splits, axis);
CHECK_VALID_OP_ATTR(SplitAttrs);

} // namespace FlexFlow

#endif
