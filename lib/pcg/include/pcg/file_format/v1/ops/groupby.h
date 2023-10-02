#ifndef _FLEXFLOW_PCG_INCLUDE_PCG_FILE_FORMAT_V1_OPS_GROUPBY_ATTRS_H
#define _FLEXFLOW_PCG_INCLUDE_PCG_FILE_FORMAT_V1_OPS_GROUPBY_ATTRS_H

#include "op-attrs/ops/groupby.h"
#include "utils/visitable.h"

namespace FlexFlow {

struct V1Group_byAttrs {
  req<int> n;
  req<float> alpha;
};
FF_VISITABLE_STRUCT(V1Group_byAttrs, n, alpha);
CHECK_IS_JSONABLE(V1Group_byAttrs);

V1Group_byAttrs to_v1(Group_byAttrs const &attrs);

} // namespace FlexFlow

#endif
