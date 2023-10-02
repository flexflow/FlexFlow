#ifndef _FLEXFLOW_PCG_INCLUDE_PCG_FILE_FORMAT_V1_OPS_REPLICATE_ATTRS_H
#define _FLEXFLOW_PCG_INCLUDE_PCG_FILE_FORMAT_V1_OPS_REPLICATE_ATTRS_H

#include "op-attrs/ff_dim.h"
#include "op-attrs/ops/replicate.h"
#include "utils/visitable.h"

namespace FlexFlow {

struct V1ReplicateAttrs {
  ff_dim_t replicate_dim;
  req<int> replicate_degree;
};
FF_VISITABLE_STRUCT(V1ReplicateAttrs, replicate_dim, replicate_degree);
CHECK_IS_JSONABLE(V1ReplicateAttrs);

V1ReplicateAttrs to_v1(ReplicateAttrs const &attrs);

} // namespace FlexFlow

#endif
