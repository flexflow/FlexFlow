#include "task-spec/dynamic_graph/parallel_op_data_movement.h"
#include "op-attrs/pcg_operator_attrs.h"
#include "utils/exception.h"

namespace FlexFlow {

bool is_parallel_op(OperatorType op_type) {
  switch (op_type) {
    case OperatorType::REPLICATE:
    case OperatorType::COMBINE:
    case OperatorType::REDUCTION:
    case OperatorType::REPARTITION:
      return true;
    default:
      return false;
  }
}

ParallelOpMovementKind
    get_parallel_op_movement_kind(OperatorType const op_type,
                                  DynamicTaskType const task_type) {
  ASSERT(is_parallel_op(op_type));
  switch (op_type) {
    case OperatorType::REPLICATE:
      switch (task_type) {
        case DynamicTaskType::FWD:
          return ParallelOpMovementKind::BROADCAST;
        case DynamicTaskType::BWD:
          return ParallelOpMovementKind::SUM_REDUCE;
        default:
          PANIC("Unexpected task type for REPLICATE", task_type);
      }
    case OperatorType::COMBINE:
      switch (task_type) {
        case DynamicTaskType::FWD:
          return ParallelOpMovementKind::GATHER;
        case DynamicTaskType::BWD:
          return ParallelOpMovementKind::BROADCAST;
        default:
          PANIC("Unexpected task type for COMBINE", task_type);
      }
    case OperatorType::REDUCTION:
      switch (task_type) {
        case DynamicTaskType::FWD:
          return ParallelOpMovementKind::SUM_REDUCE;
        case DynamicTaskType::BWD:
          return ParallelOpMovementKind::BROADCAST;
        default:
          PANIC("Unexpected task type for REDUCTION", task_type);
      }
    case OperatorType::REPARTITION:
      return ParallelOpMovementKind::RESHUFFLE;
    default:
      PANIC("Not a parallel op", op_type);
  }
}

static std::optional<OperatorType>
    try_get_pcg_op_type(TrainingOperationAttrs const &op_attrs) {
  if (!op_attrs.is_pcg_op()) {
    return std::nullopt;
  }
  return pcg_op_attrs_get_op_type(op_attrs.require_pcg_op());
}

std::optional<ParallelOpMovementKind>
    get_parallel_op_movement_kind_for_training_op(
        TrainingOperationAttrs const &op_attrs,
        DynamicTaskType const task_type) {
  std::optional<OperatorType> const op_type = try_get_pcg_op_type(op_attrs);
  if (!op_type.has_value() || !is_parallel_op(op_type.value())) {
    return std::nullopt;
  }
  return get_parallel_op_movement_kind(op_type.value(), task_type);
}

bool is_parallel_training_op(TrainingOperationAttrs const &op_attrs) {
  std::optional<OperatorType> const op_type = try_get_pcg_op_type(op_attrs);
  return op_type.has_value() && is_parallel_op(op_type.value());
}
} // namespace FlexFlow
