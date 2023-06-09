#ifndef _FLEXFLOW_RUNTIME_SRC_FUSED_OP_ATTRS_H
#define _FLEXFLOW_RUNTIME_SRC_FUSED_OP_ATTRS_H

#include "op-attrs/get_op_type.h"
#include "op-attrs/ops/core.h"
#include "op-attrs/parallel_tensor_shape.h"
#include "operator.h"
#include "utils/visitable.h"

namespace FlexFlow {

struct FusedOpAttrs : public use_visitable_cmp<FusedOpAttrs> {
  LabelledOpenMultiDiGraph<Operator, ParallelTensor> graph;
};

OperatorType get_op_type(FusedOpAttrs const &);

} // namespace FlexFlow

VISITABLE_STRUCT(::FlexFlow::FusedOpAttrs, graph);
MAKE_VISIT_HASHABLE(::FlexFlow::FusedOpAttrs);

namespace FlexFlow {
static_assert(is_valid_opattr<FusedOpAttrs>::value, "");
}

#endif
