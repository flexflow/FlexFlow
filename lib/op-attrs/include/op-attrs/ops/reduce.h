#ifndef _FLEXFLOW_OP_META_OPS_REDUCE_ATTRS_H
#define _FLEXFLOW_OP_META_OPS_REDUCE_ATTRS_H

#include "core.h"
#include "op-attrs/ff_dim.h"
#include "op-attrs/op.h"
#include "op-attrs/parallel_tensor_shape.h"
#include "utils/stack_vector.h"
#include "utils/visitable.h"

namespace FlexFlow {

struct ReduceAttrs {
  stack_vector<ff_dim_t, MAX_TENSOR_DIM> axes;
  req<Op> op_type;
  req<bool> keepdims;
};
FF_VISITABLE_STRUCT(ReduceAttrs, axes, op_type, keepdims);
CHECK_VALID_OP_ATTR(ReduceAttrs);

} // namespace FlexFlow

#endif
