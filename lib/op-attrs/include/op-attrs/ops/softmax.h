#ifndef _FLEXFLOW_SOFTMAX_ATTRS_H
#define _FLEXFLOW_SOFTMAX_ATTRS_H

#include "core.h"
#include "op-attrs/ff_dim.h"
#include "op-attrs/parallel_tensor_shape.h"
#include "utils/visitable.h"

namespace FlexFlow {

struct SoftmaxAttrs {
  ff_dim_t dim;
};
FF_VISITABLE_STRUCT(SoftmaxAttrs, dim);
CHECK_VALID_OP_ATTR(SoftmaxAttrs);

} // namespace FlexFlow

#endif
