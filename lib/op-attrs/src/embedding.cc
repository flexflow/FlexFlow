#include "op-attrs/ops/embedding.h"

namespace FlexFlow {

std::string format_as(AggregateOp const &o) {
  switch (o) {
    case AggregateOp::SUM:
      return "Sum";
    case AggregateOp::AVG:
      return "Avg";
    default:
      throw mk_runtime_error("Unknown AggregateOp with value {}",
                             static_cast<int>(o));
  }
}

} // namespace FlexFlow
