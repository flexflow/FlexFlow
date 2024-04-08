#include "op-attrs/pool_op.h"
#include "utils/exception.h"

namespace FlexFlow {

std::string format_as(PoolOp o) {
  switch (o) {
    case PoolOp::MAX:
      return "MAX";
    case PoolOp::AVG:
      return "AVG";
    default:
      throw mk_runtime_error(
          fmt::format("Unknown pool op {}", static_cast<int>(o)));
  }
}

} // namespace FlexFlow
