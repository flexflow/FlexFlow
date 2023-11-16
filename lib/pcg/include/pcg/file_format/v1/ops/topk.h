#ifndef _FLEXFLOW_PCG_INCLUDE_PCG_FILE_FORMAT_V1_OPS_TOPK_ATTRS_H
#define _FLEXFLOW_PCG_INCLUDE_PCG_FILE_FORMAT_V1_OPS_TOPK_ATTRS_H

#include "op-attrs/ops/topk.h"
#include "utils/visitable.h"

namespace FlexFlow {

struct V1TopKAttrs {
  req<int> k;
  req<bool> sorted;
};
FF_VISITABLE_STRUCT(V1TopKAttrs, k, sorted);
CHECK_IS_JSONABLE(V1TopKAttrs);

V1TopKAttrs to_v1(TopKAttrs const &a);
TopKAttrs from_v1(V1TopKAttrs const &va);

} // namespace FlexFlow

#endif
