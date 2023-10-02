#ifndef _FLEXFLOW_PCG_INCLUDE_PCG_FILE_FORMAT_V1_OPS_SOFTMAX_ATTRS_H
#define _FLEXFLOW_PCG_INCLUDE_PCG_FILE_FORMAT_V1_OPS_SOFTMAX_ATTRS_H

#include "op-attrs/ff_dim.h"
#include "op-attrs/ops/softmax.h"
#include "utils/visitable.h"

namespace FlexFlow {

struct V1SoftmaxAttrs {
  ff_dim_t dim;
};
FF_VISITABLE_STRUCT(V1SoftmaxAttrs, dim);
CHECK_IS_JSONABLE(V1SoftmaxAttrs);

V1SoftmaxAttrs to_v1(SoftmaxAttrs const &attrs);

} // namespace FlexFlow

#endif
