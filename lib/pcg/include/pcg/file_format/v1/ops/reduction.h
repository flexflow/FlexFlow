#ifndef _FLEXFLOW_PCG_INCLUDE_PCG_FILE_FORMAT_V1_OPS_REDUCTION_ATTRS_H
#define _FLEXFLOW_PCG_INCLUDE_PCG_FILE_FORMAT_V1_OPS_REDUCTION_ATTRS_H

#include "op-attrs/ops/reduction.h"
#include "pcg/file_format/v1/ff_dim.h"
#include "utils/visitable.h"

namespace FlexFlow {

struct V1ReductionAttrs {
  int reduction_dim;
  req<int> reduction_degree;
};
FF_VISITABLE_STRUCT(V1ReductionAttrs, reduction_dim, reduction_degree);
CHECK_IS_JSONABLE(V1ReductionAttrs);

V1ReductionAttrs to_v1(ReductionAttrs const &a);
ReductionAttrs from_v1(V1ReductionAttrs const &va);

} // namespace FlexFlow

#endif
