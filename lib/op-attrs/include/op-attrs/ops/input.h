#ifndef _FLEXFLOW_OP_ATTRS_OPS_OP_ATTRS_INPUT_H
#define _FLEXFLOW_OP_ATTRS_OPS_OP_ATTRS_INPUT_H

#include "core.h"
#include "utils/visitable.h"

namespace FlexFlow {

struct InputAttrs {};
FF_VISITABLE_STRUCT(InputAttrs);
CHECK_VALID_OP_ATTR(InputAttrs);

} // namespace FlexFlow

#endif
