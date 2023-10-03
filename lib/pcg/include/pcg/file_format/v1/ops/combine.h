#ifndef _FLEXFLOW_PCG_INCLUDE_PCG_FILE_FORMAT_V1_OPS_COMBINE_ATTRS_H
#define _FLEXFLOW_PCG_INCLUDE_PCG_FILE_FORMAT_V1_OPS_COMBINE_ATTRS_H

#include "op-attrs/ops/combine.h"
#include "utils/visitable.h"

namespace FlexFlow {

struct V1CombineAttrs {
  int combine_dim;
  req<int> combine_degree;
};
FF_VISITABLE_STRUCT(V1CombineAttrs, combine_dim, combine_degree);
CHECK_IS_JSONABLE(V1CombineAttrs);

V1CombineAttrs to_v1(CombineAttrs const &attrs);

} // namespace FlexFlow

#endif
