#include "op-attrs/pcg_operator_attrs.h"
#include "op-attrs/get_op_type.h"
#include "op-attrs/ops/linear.h"
#include "utils/overload.h"

namespace FlexFlow {

bool is_parallel_op(PCGOperatorAttrs const &attrs) {
  return (attrs.has<CombineAttrs>() || attrs.has<ReductionAttrs>() ||
          attrs.has<RepartitionAttrs>() || attrs.has<ReplicateAttrs>());
}

OperatorType get_op_type(PCGOperatorAttrs const &attrs) {
  return attrs.visit<OperatorType>(
      [](auto const &x) { return get_op_type(x); });
}

RecordFormatter as_dot(PCGOperatorAttrs const &attrs) {
  return attrs.visit<RecordFormatter>(overload{
      [](LinearAttrs const &l) { return as_dot(l); },
      [&](auto const &) {
        RecordFormatter r;
        r << fmt::to_string(get_op_type(attrs));
        return r;
      },
  });
}

PCGOperatorAttrs pcg_op_attrs_from_compgraph_op_attrs(ComputationGraphOpAttrs const &cg_attrs) {
  return cg_attrs.visit<PCGOperatorAttrs>(overload {
    [](auto const &attrs) { return PCGOperatorAttrs{attrs}; },
  });
}

} // namespace FlexFlow
