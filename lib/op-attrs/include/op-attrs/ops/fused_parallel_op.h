#ifndef _FLEXFLOW_FUSED_PARALLEL_OP_ATTRS_H
#define _FLEXFLOW_FUSED_PARALLEL_OP_ATTRS_H

#include "op-attrs/ops/core.h"
#include "op-attrs/parallel_tensor_shape.h"
#include "parallel_op_info.h"
#include "utils/visitable.h"

namespace FlexFlow {

struct FusedParallelOpAttrs : public use_visitable_cmp<FusedParallelOpAttrs> {
public:
  FusedParallelOpAttrs() = delete;
  explicit FusedParallelOpAttrs(
      stack_vector<ParallelOpInfo, MAX_NUM_FUSED_OPERATORS> const &);

public:
  stack_vector<ParallelOpInfo, MAX_NUM_FUSED_OPERATORS> parallel_ops;
};

} // namespace FlexFlow

VISITABLE_STRUCT(::FlexFlow::FusedParallelOpAttrs, parallel_ops);
MAKE_VISIT_HASHABLE(::FlexFlow::FusedParallelOpAttrs);

namespace FlexFlow {
static_assert(is_valid_opattr<FusedParallelOpAttrs>::value, "");
}

#endif
