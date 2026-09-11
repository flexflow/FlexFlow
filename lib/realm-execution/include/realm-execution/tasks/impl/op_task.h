#ifndef _FLEXFLOW_LIB_REALM_EXECUTION_INCLUDE_REALM_EXECUTION_TASKS_IMPL_OP_TASK_H
#define _FLEXFLOW_LIB_REALM_EXECUTION_INCLUDE_REALM_EXECUTION_TASKS_IMPL_OP_TASK_H

#include "op-attrs/ops/loss_functions/loss_attrs.dtg.h"
#include "pcg/optimizer_attrs.dtg.h"
#include "realm-execution/device_specific_managed_per_device_ff_handle.h"
#include "realm-execution/device_specific_ptr.h"
#include "realm-execution/realm.h"
#include "realm-execution/realm_context.h"
#include "realm-execution/tensor_instance_backing.dtg.h"
#include "task-spec/dynamic_graph/dynamic_node_invocation.dtg.h"
#include "task-spec/per_device_op_state.dtg.h"
#include <optional>

namespace FlexFlow {

/**
 * \brief The function registered as a Realm task for operator-related tasks.
 * Dispatched by \ref spawn_op_task.
 */
void op_task_body(void const *, size_t, void const *, size_t, Realm::Processor);

/**
 * \brief Launches the task (\ref op_task_body), for a \ref
 * DynamicNodeInvocation using Realm.
 *
 * The task launch process functions a bit differently to that used in the
 * previous FlexFlow codebase. Rather than having a function registered with
 * realm/legion for every \ref task_id_t, we now have only a few functions
 * registered: \ref op_task_body, \ref ff_handle_init_task_body,
 * \ref per_device_op_state_init_return_task_body, and \ref controller_task_body
 * (see \ref register_all_tasks for where this list comes from), and in fact
 * only \ref op_task_body is launched by \ref spawn_op_task. Each of these
 * registered tasks use the serialized arguments sent to them to dispatch to the
 * correct implementatin in task-spec: for example, if we are trying to launch
 * the task for a Conv2d operator, this function will actually dispatch a call
 * to \ref op_task_body with a serialized \ref OpTaskArgs as an argument, and
 * then \ref op_task_body will deserialize the argument, determine that we are
 * trying to launch the forward pass of Conv2d, use \ref
 * execute_dynamic_node_invocation (which then uses \ref call_fwd_task_impl) to
 * actually call the function in lib/task-spec/src/task-spec/ops/impl/conv_2d.cc
 *
 * The above also means that we don't have a separate
 * \ref ITaskArgumentAccessor subclass for realm-execution. Instead we ship over
 * the information on the corresponding realm instances over to the remote node,
 * grab the corresponding pointer/\ref GenericTensorAccessor, and then use
 * \ref LocalTaskArgumentAccessor for the actual argument access as, by this
 * point, everything is local.
 *
 * To understand how this fits into the broader structure of \ref
 * realm-execution, see \ref realm-execution-tasks.
 */
Realm::Event spawn_op_task(
    RealmContext &ctx,
    Realm::Processor target_proc,
    DynamicNodeInvocation const &invocation,
    TensorInstanceBacking const &tensor_backing,
    std::optional<DeviceSpecificPtr<PerDeviceOpState>> const &device_state,
    DeviceSpecificPtr<ManagedPerDeviceFFHandle> const &device_handle,
    std::optional<OptimizerAttrs> const &optimizer_attrs,
    Realm::Event precondition);

} // namespace FlexFlow

#endif
