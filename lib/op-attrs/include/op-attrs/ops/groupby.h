#ifndef _FLEXFLOW_GROUPBY_ATTRS_H
#define _FLEXFLOW_GROUPBY_ATTRS_H

#include "core.h"
#include "op-attrs/parallel_tensor_shape.h"
#include "utils/visitable.h"

namespace FlexFlow {

struct Group_byAttrs {
  int n;
  req<float> alpha;
};
FF_VISITABLE_STRUCT(Group_byAttrs, n, alpha);
FF_VISIT_FMTABLE(Group_byAttrs);

CHECK_VALID_OP_ATTR(Group_byAttrs);

} // namespace FlexFlow

#endif
