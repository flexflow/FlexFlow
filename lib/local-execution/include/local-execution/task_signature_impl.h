#ifndef _FLEXFLOW_LOCAL_EXECUTION_TASK_SIGNATURE_IMPL_H
#define _FLEXFLOW_LOCAL_EXECUTION_TASK_SIGNATURE_IMPL_H

// #include "local-execution/device_specific.h"
// #include "local-execution/device_states.h"
#include "local-execution/op_task_invocation.h"
#include "local-execution/task_impl_function.dtg.h"
#include "local-execution/tasks.h"
#include "op-attrs/computation_graph_op_attrs.h"
// #include "task_argument_accessor.h"
#include "utils/variant.h"

namespace FlexFlow {

// using TaskImplFunction = std::variant<
//     std::function<DeviceSpecific<DeviceStates>(TaskArgumentAccessor const
//     &)>, std::function<std::optional<float>(TaskArgumentAccessor const &)>>;

struct TaskSignatureAndImpl {
  TaskImplFunction impl_function;
  OpTaskSignature task_signature;

  bool operator==(TaskSignatureAndImpl const &other) const;
  bool operator!=(TaskSignatureAndImpl const &other) const;
};

TaskSignatureAndImpl get_task_sig_impl(task_id_t const &);
std::vector<task_id_t> get_task_ids(ComputationGraphOpAttrs const &);

OpTaskInvocation init(ComputationGraphOpAttrs const &);
OpTaskInvocation forward(ComputationGraphOpAttrs const &);
OpTaskInvocation backward(ComputationGraphOpAttrs const &);

} // namespace FlexFlow

#endif
