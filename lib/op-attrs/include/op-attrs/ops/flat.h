#ifndef _FLEXFLOW_FLAT_ATTRS_H
#define _FLEXFLOW_FLAT_ATTRS_H

#include "core.h"
#include "op-attrs/parallel_tensor_shape.h"
#include "utils/visitable.h"

namespace FlexFlow {

struct FlatAttrs {};
FF_VISITABLE_STRUCT(FlatAttrs);
CHECK_VALID_OP_ATTR(FlatAttrs);

} // namespace FlexFlow

#endif
