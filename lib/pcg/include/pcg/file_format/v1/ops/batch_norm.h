#ifndef _FLEXFLOW_PCG_INCLUDE_PCG_FILE_FORMAT_V1_OPS_BATCH_NORM_H
#define _FLEXFLOW_PCG_INCLUDE_PCG_FILE_FORMAT_V1_OPS_BATCH_NORM_H

#include "op-attrs/ops/batch_norm.h"
#include "utils/visitable.h"

namespace FlexFlow {

struct V1BatchNormAttrs {
  req<bool> relu;
};
FF_VISITABLE_STRUCT(V1BatchNormAttrs, relu);
CHECK_IS_JSONABLE(V1BatchNormAttrs);

V1BatchNormAttrs to_v1(BatchNormAttrs const &attrs);

} // namespace FlexFlow

#endif
