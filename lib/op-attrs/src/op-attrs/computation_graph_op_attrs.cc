#include "op-attrs/computation_graph_op_attrs.h"
#include "op-attrs/get_op_type.h"
#include "utils/overload.h"

namespace FlexFlow {

OperatorType get_op_type(ComputationGraphOpAttrs const &attrs) {
  return attrs.visit<OperatorType>(
      [](auto const &x) { return get_op_type(x); });
}

ComputationGraphOpAttrs
    compgraph_op_attrs_from_pcg_op_attrs(PCGOperatorAttrs const &op) {
  auto fail_on_parallel_op = [](auto const &attrs) -> ComputationGraphOpAttrs {
    throw mk_runtime_error(
        fmt::format("Encountered parallel operator in "
                    "compgraph_op_attrs_from_pcg_op_attrs: {}",
                    attrs));
  };

  return op.visit<ComputationGraphOpAttrs>(overload{
      [&](CombineAttrs const &attrs) { return fail_on_parallel_op(attrs); },
      [&](ReductionAttrs const &attrs) { return fail_on_parallel_op(attrs); },
      [&](RepartitionAttrs const &attrs) { return fail_on_parallel_op(attrs); },
      [&](ReplicateAttrs const &attrs) { return fail_on_parallel_op(attrs); },
      [](auto const &attrs) { return ComputationGraphOpAttrs{attrs}; },
  });
}

} // namespace FlexFlow
