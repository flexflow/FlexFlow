#ifndef _FLEXFLOW_PCG_INCLUDE_PCG_FILE_FORMAT_V1_OPS_SPLIT_ATTRS_H
#define _FLEXFLOW_PCG_INCLUDE_PCG_FILE_FORMAT_V1_OPS_SPLIT_ATTRS_H

#include "op-attrs/ops/split.h"
#include "utils/visitable.h"

namespace FlexFlow {

struct V1SplitAttrs {
  // The size of this vector must be <= MAX_TENSOR_DIM
  std::vector<int> splits;
  req<int> axis;
};
FF_VISITABLE_STRUCT(V1SplitAttrs, splits, axis);
CHECK_IS_JSONABLE(V1SplitAttrs);

V1SplitAttrs to_v1(SplitAttrs const &a);
SplitAttrs from_v1(V1SplitAttrs const &va);

} // namespace FlexFlow

#endif
