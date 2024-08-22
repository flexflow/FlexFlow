
#ifndef _FLEXFLOW_LOCAL_EXECUTION_TASK_REGISTRY_H
#define _FLEXFLOW_LOCAL_EXECUTION_TASK_REGISTRY_H

#include "local-execution/task_registry.dtg.h"
#include "op-attrs/operator_attrs.h"

namespace FlexFlow {

void register_tasks_for_layer(TaskRegistry &,
                              layer_guid_t const &,
                              ComputationGraphOpAttrs const &attrs);

} // namespace FlexFlow

#endif
