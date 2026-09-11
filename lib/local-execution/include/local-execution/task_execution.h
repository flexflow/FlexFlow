#ifndef _FLEXFLOW_LIB_LOCAL_EXECUTION_INCLUDE_LOCAL_EXECUTION_TASK_EXECUTION_H
#define _FLEXFLOW_LIB_LOCAL_EXECUTION_INCLUDE_LOCAL_EXECUTION_TASK_EXECUTION_H

#include "kernels/profiling_settings.dtg.h"
#include "task-spec/dynamic_graph/dynamic_node_invocation.dtg.h"
#include "task-spec/per_device_op_state.dtg.h"
#include "task-spec/task_argument_accessor/task_argument_accessor.h"
#include "utils/units/milliseconds_t.h"

namespace FlexFlow {

TaskArgumentAccessor make_task_argument_accessor_for_invocation(
    DynamicNodeInvocation const &invocation,
    Allocator &allocator,
    std::optional<ProfilingSettings> const &profiling_settings,
    device_handle_t const &ff_handle,
    std::optional<PerDeviceOpState> const &per_device_op_state,
    std::optional<OptimizerAttrs> const &optimizer_attrs,
    global_device_id_t device_idx);

std::optional<milliseconds_t> execute_dynamic_node_invocation(
    DynamicNodeInvocation const &invocation,
    Allocator &allocator,
    std::optional<ProfilingSettings> const &profiling_settings,
    device_handle_t const &ff_handle,
    std::optional<PerDeviceOpState> const &per_device_op_state,
    std::optional<OptimizerAttrs> const &optimizer_attrs,
    global_device_id_t device_idx);

} // namespace FlexFlow

#endif
