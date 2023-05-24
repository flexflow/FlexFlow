#ifndef _FLEXFLOW_RUNTIME_SRC_FUSED_OP_ATTRS_H
#define _FLEXFLOW_RUNTIME_SRC_FUSED_OP_ATTRS_H

#include "op-attrs/parallel_tensor_shape.h"
#include "utils/visitable.h"
#include "op-attrs/ops/core.h"
#include "operator.h"
#include "op-attrs/get_op_type.h"

namespace FlexFlow {

struct FusedOpAttrs : public use_visitable_cmp<FusedOpAttrs> {
  LabelledOpenMultiDiGraph<Operator, ParallelTensor> graph;
};

OperatorType get_op_type(FusedOpAttrs const &);

}

VISITABLE_STRUCT(::FlexFlow::FusedOpAttrs, graph);
MAKE_VISIT_HASHABLE(::FlexFlow::FusedOpAttrs);

namespace FlexFlow {
static_assert(is_valid_opattr<FusedOpAttrs>::value, "");
}

#endif 
