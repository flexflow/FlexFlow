#ifndef _FLEXFLOW_PCG_INCLUDE_PCG_FILE_FORMAT_V1_OPS_NOOP_H
#define _FLEXFLOW_PCG_INCLUDE_PCG_FILE_FORMAT_V1_OPS_NOOP_H

#include "op-attrs/ops/noop.h"
#include "utils/json.h"
#include "utils/visitable.h"

namespace FlexFlow {

struct V1NoopAttrs {};
FF_VISITABLE_STRUCT(V1NoopAttrs);
CHECK_IS_JSONABLE(NoopAttrs);

V1NoopAttrs to_v1(NoopAttrs const &a);
NoopAttrs from_v1(V1NoopAttrs const &va);

} // namespace FlexFlow

#endif
