#ifndef _FLEXFLOW_PARTITION_ATTRS_H
#define _FLEXFLOW_PARTITION_ATTRS_H

#include "core.h"
#include "op-attrs/ops/repartition_attrs.h"
#include "op-attrs/parallel_tensor_shape.h"

namespace FlexFlow {

CHECK_VALID_OP_ATTR(RepartitionAttrs);

} // namespace FlexFlow

#endif
