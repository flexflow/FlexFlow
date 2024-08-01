#ifndef _FLEXFLOW_LOCAL_EXECUTION_TASK_SIGNATURE_IMPL_H
#define _FLEXFLOW_LOCAL_EXECUTION_TASK_SIGNATURE_IMPL_H

#include "local-execution/op_task_invocation.h"
#include "local-execution/task_impl_function.dtg.h"
#include "local-execution/tasks.h"
#include "op-attrs/computation_graph_op_attrs.h"
#include "utils/variant.h"

namespace FlexFlow {

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

std::string format_as(TaskSignatureAndImpl const &x);
std::ostream &operator<<(std::ostream &s, TaskSignatureAndImpl const &x);

} // namespace FlexFlow

#endif
