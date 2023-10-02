#ifndef _FLEXFLOW_PCG_INCLUDE_PCG_FILE_FORMAT_V1_OPS_INPUT_H
#define _FLEXFLOW_PCG_INCLUDE_PCG_FILE_FORMAT_V1_OPS_INPUT_H

#include "op-attrs/ops/input.h"
#include "utils/visitable.h"

namespace FlexFlow {

struct V1InputAttrs {};
FF_VISITABLE_STRUCT(V1InputAttrs);
CHECK_IS_JSONABLE(V1InputAttrs);

V1InputAttrs to_v1(InputAttrs const &attrs);

} // namespace FlexFlow

#endif
