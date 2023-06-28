#ifndef _FLEXFLOW_PARTITION_ATTRS_H
#define _FLEXFLOW_PARTITION_ATTRS_H

#include "core.h"
#include "op-attrs/ff_dim.h"
#include "op-attrs/parallel_tensor_shape.h"
#include "utils/visitable.h"

namespace FlexFlow {

struct RepartitionAttrs {
  ff_dim_t repartition_dim;
  req<int> repartition_degree;
};
FF_VISITABLE_STRUCT(RepartitionAttrs, repartition_dim, repartition_degree);
CHECK_VALID_OP_ATTR(RepartitionAttrs);

} // namespace FlexFlow

#endif
