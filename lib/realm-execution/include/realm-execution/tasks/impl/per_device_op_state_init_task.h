#ifndef _FLEXFLOW_LIB_REALM_EXECUTION_INCLUDE_REALM_EXECUTION_TASKS_IMPL_PER_DEVICE_OP_STATE_INIT_TASK_H
#define _FLEXFLOW_LIB_REALM_EXECUTION_INCLUDE_REALM_EXECUTION_TASKS_IMPL_PER_DEVICE_OP_STATE_INIT_TASK_H

#include "pcg/optimizer_attrs.dtg.h"
#include "realm-execution/device_specific_managed_per_device_ff_handle.h"
#include "realm-execution/device_specific_ptr.h"
#include "realm-execution/realm.h"
#include "realm-execution/realm_context.h"
#include "realm-execution/tensor_instance_backing.dtg.h"
#include "task-spec/dynamic_graph/dynamic_node_invocation.dtg.h"
#include "task-spec/per_device_op_state.dtg.h"

namespace FlexFlow {

/**
 * \brief The function registered as a Realm task for starting the asynchronous
 * initialization of the \ref PerDeviceOpState. Dispatched by \ref
 * spawn_per_device_op_state_init_task.
 *
 * To understand how this fits into the broader structure of \ref
 * realm-execution, see \ref realm-execution-tasks.
 */
void per_device_op_state_init_task_body(
    void const *, size_t, void const *, size_t, Realm::Processor);

/**
 * \brief Launches the task (\ref per_device_op_state_init_task_body) for
 * starting the asynchronous initialization of the \ref PerDeviceOpState.
 *
 * To understand how this fits into the broader structure of \ref
 * realm-execution, see \ref realm-execution-tasks.
 */
std::optional<Realm::Event> spawn_per_device_op_state_init_task(
    RealmContext &ctx,
    Realm::Processor target_proc,
    DynamicNodeInvocation const &invocation,
    TensorInstanceBacking const &tensor_backing,
    DeviceSpecificPtr<ManagedPerDeviceFFHandle> const &device_handle,
    OptimizerAttrs const &optimizer_attrs,
    DeviceSpecificPtr<PerDeviceOpState> *result_ptr,
    Realm::Event precondition);

} // namespace FlexFlow

#endif
