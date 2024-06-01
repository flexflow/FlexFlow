#ifndef _FLEXFLOW_GATHER_ATTRS_H
#define _FLEXFLOW_GATHER_ATTRS_H

#include "core.h"
#include "op-attrs/ff_dim.h"
#include "op-attrs/parallel_tensor_shape.h"
#include "utils/visitable.h"

namespace FlexFlow {

struct GatherAttrs {
  req<ff_dim_t> ff_dim;
};
FF_VISITABLE_STRUCT(GatherAttrs, ff_dim);
CHECK_VALID_OP_ATTR(GatherAttrs);

} // namespace FlexFlow

#endif
