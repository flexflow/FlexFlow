#ifndef _FLEXFLOW_OP_ATTRS_OPS_NOOP_H
#define _FLEXFLOW_OP_ATTRS_OPS_NOOP_H

#include "core.h"
#include "utils/visitable.h"

namespace FlexFlow {

struct NoopAttrs {};
FF_VISITABLE_STRUCT(NoopAttrs);
CHECK_VALID_OP_ATTR(NoopAttrs);

} // namespace FlexFlow

#endif
