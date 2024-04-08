#include "op-attrs/aggregate_op.h"
#include "utils/exception.h"

namespace FlexFlow {

std::string format_as(AggregateOp o) {
  switch (o) {
    case AggregateOp::SUM:
      return "SUM";
    case AggregateOp::AVG:
      return "AVG";
    default:
      throw mk_runtime_error(fmt::format("Unknown aggregate op {}", static_cast<int>(o)));
  }
}

}
