
#ifndef _FLEXFLOW_LOCAL_EXECUTION_TASK_REGISTRY_H
#define _FLEXFLOW_LOCAL_EXECUTION_TASK_REGISTRY_H

#include "local-execution/op_task_invocation.h"
#include "local-execution/task_signature_impl.h"
#include "op-attrs/operator_attrs.h"
#include "pcg/computation_graph.h"

namespace FlexFlow {

struct TaskRegistry {
  TaskRegistry() = default;

  void register_task(task_id_t const &,
                     operator_guid_t const &,
                     CompGraphOperatorAttrs const &attrs);

  std::unordered_map<operator_guid_t, task_id_t> init_task_ids;
  std::unordered_map<operator_guid_t, task_id_t> forward_task_ids;
  std::unordered_map<operator_guid_t, task_id_t> backward_task_ids;
  std::unordered_map<task_id_t, TaskSignatureAndImpl> task_mapping;
};

} // namespace FlexFlow

#endif
