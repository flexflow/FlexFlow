#ifndef _FLEXFLOW_OP_ATTRS_INCLUDE_OP_ATTRS_OPS_REVERSE_H
#define _FLEXFLOW_OP_ATTRS_INCLUDE_OP_ATTRS_OPS_REVERSE_H

#include "core.h"
#include "op-attrs/ff_dim.h"
#include "utils/visitable.h"

namespace FlexFlow {

struct ReverseAttrs {
  ff_dim_t axis;
};
FF_VISITABLE_STRUCT(ReverseAttrs, axis);
CHECK_VALID_OP_ATTR(ReverseAttrs);

} // namespace FlexFlow

#endif
