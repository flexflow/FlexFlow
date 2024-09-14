#include "op-attrs/computation_graph_op_attrs.h"
#include "op-attrs/get_op_type.h"
#include "op-attrs/ops/broadcast.h"
#include "op-attrs/ops/linear.h"
#include "utils/overload.h"

namespace FlexFlow {

OperatorType get_op_type(ComputationGraphOpAttrs const &attrs) {
  return attrs.visit<OperatorType>(
      [](auto const &x) { return get_op_type(x); });
}

RecordFormatter as_dot(ComputationGraphOpAttrs const &attrs) {
  return attrs.visit<RecordFormatter>(overload{
      [](LinearAttrs const &l) { return as_dot(l); },
      [](BroadcastAttrs const &a) { return as_dot(a); },
      [&](auto const &) {
        RecordFormatter r;
        r << fmt::to_string(get_op_type(attrs));
        return r;
      },
  });
}

} // namespace FlexFlow
