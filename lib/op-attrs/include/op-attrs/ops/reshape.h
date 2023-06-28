#ifndef _FLEXFLOW_RESHAPE_ATTRS_H
#define _FLEXFLOW_RESHAPE_ATTRS_H

#include "core.h"
#include "op-attrs/tensor_shape.h"
#include "utils/visitable.h"

namespace FlexFlow {

struct ReshapeAttrs {
  TensorShape shape;
};
FF_VISITABLE_STRUCT(ReshapeAttrs, shape);
CHECK_VALID_OP_ATTR(ReshapeAttrs);

} // namespace FlexFlow

#endif
