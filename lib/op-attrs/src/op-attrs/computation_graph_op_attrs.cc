#include "op-attrs/computation_graph_op_attrs.h"
#include "op-attrs/get_op_type.h"

namespace FlexFlow {

OperatorType get_op_type(ComputationGraphOpAttrs const &attrs) {
  return attrs.visit<OperatorType>(
      [](auto const &x) { return get_op_type(x); });
}

} // namespace FlexFlow
