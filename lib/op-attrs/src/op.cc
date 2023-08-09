#include "op-attrs/op.h"

namespace FlexFlow {

std::string get_operator_type_name(Op op) {
  return fmt::to_string(op);
}

bool is_parallel_op(OperatorType const &t) {
  switch (t) {
    case Op::REPARTITION:
    case Op::COMBINE:
    case Op::REPLICATE:
    case Op::REDUCTION:
    case Op::BATCH:
    case Op::PIPELINE:
    case Op::FUSED_PARALLEL:
      return true;
    default:
      return false;
  }
}

} // namespace FlexFlow
