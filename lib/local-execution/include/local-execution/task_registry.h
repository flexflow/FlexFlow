
#ifndef _FLEXFLOW_LOCAL_EXECUTION_TASK_REGISTRY_H
#define _FLEXFLOW_LOCAL_EXECUTION_TASK_REGISTRY_H

#include "local-execution/op_task_type.dtg.h"
#include "local-execution/task_registry.dtg.h"
#include "op-attrs/computation_graph_op_attrs.h"

namespace FlexFlow {

TaskRegistry empty_task_registry();

void register_tasks_for_layer(TaskRegistry &,
                              layer_guid_t const &,
                              ComputationGraphOpAttrs const &attrs);

bool registry_contains_op_task(TaskRegistry const &,
                               layer_guid_t const &,
                               OpTaskType const &);

} // namespace FlexFlow

#endif
