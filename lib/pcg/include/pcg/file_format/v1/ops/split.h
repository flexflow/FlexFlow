#ifndef _FLEXFLOW_PCG_INCLUDE_PCG_FILE_FORMAT_V1_OPS_SPLIT_ATTRS_H
#define _FLEXFLOW_PCG_INCLUDE_PCG_FILE_FORMAT_V1_OPS_SPLIT_ATTRS_H

#include "op-attrs/ff_dim.h"
#include "op-attrs/ops/split.h"
#include "utils/visitable.h"

namespace FlexFlow {

struct V1SplitAttrs {
  req<stack_vector<int, MAX_NUM_OUTPUTS>> splits;
  ff_dim_t axis;
};
FF_VISITABLE_STRUCT(V1SplitAttrs, splits, axis);
CHECK_IS_JSONABLE(V1SplitAttrs);

V1SplitAttrs to_v1(SplitAttrs const &attrs);

} // namespace FlexFlow

#endif
