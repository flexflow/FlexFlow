#ifndef _FLEXFLOW_OP_META_OPS_REDUCE_ATTRS_H
#define _FLEXFLOW_OP_META_OPS_REDUCE_ATTRS_H

#include "op-attrs/op.h"
#include "op-attrs/parallel_tensor_shape.h"
#include "utils/visitable.h"
#include "utils/stack_vector.h"
#include "op-attrs/ff_dim.h"

namespace FlexFlow {

struct ReduceAttrs : public use_visitable_cmp<ReduceAttrs> {
public:
  ReduceAttrs(stack_vector<ff_dim_t, MAX_TENSOR_DIM> const &axes, Op op_type, bool keepdims);
public:
  stack_vector<ff_dim_t, MAX_TENSOR_DIM> axes;
  Op op_type;
  bool keepdims;
};

}

VISITABLE_STRUCT(::FlexFlow::ReduceAttrs, axes, keepdims);
MAKE_VISIT_HASHABLE(::FlexFlow::ReduceAttrs);

#endif
