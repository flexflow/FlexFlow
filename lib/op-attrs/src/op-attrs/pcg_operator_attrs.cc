#include "op-attrs/pcg_operator_attrs.h"
#include "op-attrs/get_op_type.h"

namespace FlexFlow {

bool is_parallel_op(PCGOperatorAttrs const &attrs) {
  return (attrs.has<CombineAttrs>() || attrs.has<ReductionAttrs>() ||
          attrs.has<RepartitionAttrs>() || attrs.has<ReplicateAttrs>());
}

OperatorType get_op_type(PCGOperatorAttrs const &attrs) {
  return attrs.visit<OperatorType>(
      [](auto const &x) { return get_op_type(x); });
}

ComputationGraphOpAttrs get_cg_attrs(PCGOperatorAttrs const &op) {
  if (is_parallel_op(op)) {
    throw mk_runtime_error(
        fmt::format("Cannot convert parallel op to non-parallel"));
  }

  return std::visit(
      [](auto &&arg) -> ComputationGraphOpAttrs {
        return ComputationGraphOpAttrs{std::forward<decltype(arg)>(arg)};
      },
      op);
}

} // namespace FlexFlow
