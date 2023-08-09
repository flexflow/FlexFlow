#ifndef _FLEXFLOW_TOPK_ATTRS_H
#define _FLEXFLOW_TOPK_ATTRS_H

#include "core.h"
#include "op-attrs/parallel_tensor_shape.h"
#include "utils/visitable.h"

namespace FlexFlow {

struct TopKAttrs {
  req<int> k;
  req<bool> sorted;
};
FF_VISITABLE_STRUCT(TopKAttrs, k, sorted);
CHECK_VALID_OP_ATTR(TopKAttrs);

} // namespace FlexFlow

#endif
