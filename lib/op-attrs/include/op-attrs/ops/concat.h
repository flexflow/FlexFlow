#ifndef _FLEXFLOW_CONCAT_ATTRS_H
#define _FLEXFLOW_CONCAT_ATTRS_H

#include "core.h"
#include "op-attrs/ff_dim.h"
#include "op-attrs/parallel_tensor_shape.h"
#include "utils/visitable.h"

namespace FlexFlow {

struct ConcatAttrs {
  ff_dim_t axis;
};
FF_VISITABLE_STRUCT(ConcatAttrs, axis);
CHECK_VALID_OP_ATTR(ConcatAttrs);

} // namespace FlexFlow

#endif
