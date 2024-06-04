#include "op-attrs/operator_type.h"

namespace FlexFlow {

std::string get_operator_type_name(OperatorType op) {
  return fmt::to_string(op);
}

bool is_parallel_op(OperatorType const &t) {
  switch (t) {
    case OperatorType::REPARTITION:
    case OperatorType::COMBINE:
    case OperatorType::REPLICATE:
    case OperatorType::REDUCTION:
    case OperatorType::BATCH:
    case OperatorType::PIPELINE:
    case OperatorType::FUSED_PARALLEL:
      return true;
    default:
      return false;
  }
}

} // namespace FlexFlow
