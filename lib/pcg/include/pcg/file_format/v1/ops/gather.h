#ifndef _FLEXFLOW_PCG_INCLUDE_PCG_FILE_FORMAT_V1_OPS_GATHER_ATTRS_H
#define _FLEXFLOW_PCG_INCLUDE_PCG_FILE_FORMAT_V1_OPS_GATHER_ATTRS_H

#include "op-attrs/ops/gather.h"
#include "utils/visitable.h"

namespace FlexFlow {

struct V1GatherAttrs {
  req<int> dim;
};
FF_VISITABLE_STRUCT(V1GatherAttrs, dim);
CHECK_IS_JSONABLE(V1GatherAttrs);

V1GatherAttrs to_v1(GatherAttrs const &a);
GatherAttrs from_v1(V1GatherAttrs const &va);

} // namespace FlexFlow

#endif
