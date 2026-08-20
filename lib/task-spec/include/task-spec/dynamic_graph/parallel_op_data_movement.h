#ifndef _FLEXFLOW_PARALLEL_OP_DATA_MOVEMENT_H
#define _FLEXFLOW_PARALLEL_OP_DATA_MOVEMENT_H
#include "op-attrs/operator_type.dtg.h"
#include "task-spec/dynamic_graph/dynamic_task_type.dtg.h"
#include "task-spec/dynamic_graph/training_operation_attrs.dtg.h"
#include <optional>

namespace FlexFlow {

enum class ParallelOpMovementKind {
  BROADCAST,  // 1 → N (copy)
  GATHER,     // N → 1 (copy)
  SUM_REDUCE, // N → 1 (sum)
  RESHUFFLE,  // N → N (copy)
};

// Returns the data movement kind for a parallel op in a given pass direction.
// This is the single source of truth that all three pipeline stages
// (pass_expansion, shard_expansion, pcg_instance) derive their behavior from.
ParallelOpMovementKind get_parallel_op_movement_kind(OperatorType op_type,
                                                     DynamicTaskType task_type);

// Returns true if this op type is a parallel (data-movement-only) op.
bool is_parallel_op(OperatorType op_type);

// Returns the movement kind if op_attrs is a parallel op, nullopt otherwise.
// Avoids requiring pcg_operator_attrs.h at call sites.
std::optional<ParallelOpMovementKind>
    get_parallel_op_movement_kind_for_training_op(
        TrainingOperationAttrs const &op_attrs, DynamicTaskType task_type);

// Returns true if the TrainingOperationAttrs represents a parallel op.
bool is_parallel_training_op(TrainingOperationAttrs const &op_attrs);
} // namespace FlexFlow

#endif
