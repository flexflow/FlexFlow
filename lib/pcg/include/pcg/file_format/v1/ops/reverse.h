#ifndef _FLEXFLOW_PCG_INCLUDE_PCG_FILE_FORMAT_V1_OPS_REVERSE_H
#define _FLEXFLOW_PCG_INCLUDE_PCG_FILE_FORMAT_V1_OPS_REVERSE_H

#include "op-attrs/ff_dim.h"
#include "op-attrs/ops/reverse.h"
#include "utils/visitable.h"

namespace FlexFlow {

struct V1ReverseAttrs {
  ff_dim_t axis;
};
FF_VISITABLE_STRUCT(V1ReverseAttrs, axis);
CHECK_IS_JSONABLE(V1ReverseAttrs);

V1ReverseAttrs to_v1(ReverseAttrs const &);

} // namespace FlexFlow

#endif
