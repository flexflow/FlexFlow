#ifndef _FLEXFLOW_DROPOUT_ATTRS_H
#define _FLEXFLOW_DROPOUT_ATTRS_H

#include "core.h"
#include "op-attrs/ops/dropout_attrs.h"
#include "op-attrs/parallel_tensor_shape.h"

namespace FlexFlow {

CHECK_VALID_OP_ATTR(DropoutAttrs);

} // namespace FlexFlow

#endif
