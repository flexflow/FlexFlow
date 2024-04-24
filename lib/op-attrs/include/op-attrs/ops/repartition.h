#ifndef _FLEXFLOW_PARTITION_ATTRS_H
#define _FLEXFLOW_PARTITION_ATTRS_H

#include "core.h"
#include "op-attrs/ops/repartition_attrs.dtg.h"
#include "op-attrs/parallel_tensor_shape.dtg.h"

namespace FlexFlow {

CHECK_VALID_OP_ATTR(RepartitionAttrs);

ParallelTensorShape get_output_shape(RepartitionAttrs const &, ParallelTensorShape const &input_shape);

} // namespace FlexFlow

#endif
