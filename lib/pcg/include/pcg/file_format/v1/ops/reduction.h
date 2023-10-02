#ifndef _FLEXFLOW_PCG_INCLUDE_PCG_FILE_FORMAT_V1_OPS_REDUCTION_ATTRS_H
#define _FLEXFLOW_PCG_INCLUDE_PCG_FILE_FORMAT_V1_OPS_REDUCTION_ATTRS_H

#include "op-attrs/ff_dim.h"
#include "op-attrs/ops/reduction.h"
#include "utils/visitable.h"

namespace FlexFlow {

struct V1ReductionAttrs {
  ff_dim_t reduction_dim;
  req<int> reduction_degree;
};
FF_VISITABLE_STRUCT(V1ReductionAttrs, reduction_dim, reduction_degree);
CHECK_IS_JSONABLE(V1ReductionAttrs);

V1ReductionAttrs to_v1(ReductionAttrs const &attrs);

} // namespace FlexFlow

#endif
