#ifndef _FLEXFLOW_RUNTIME_SRC_EXECUTABLE_TASK_INVOCATION_H
#define _FLEXFLOW_RUNTIME_SRC_EXECUTABLE_TASK_INVOCATION_H

#include "task_invocation.h"

namespace FlexFlow {

using ExecutableArgSpec = variant<ConcreteArgSpec, IndexArgSpec, CheckedTypedFuture, CheckedTypedFutureMap>;

struct ExecutableTaskBinding {
private:
  std::unordered_map<slot_id, ExecutableArgSpec> arg_bindings;
  std::unordered_map<slot_id, parallel_tensor_guid_t> bindings;
};

struct ExecutableTaskInvocation : public use_visitable_cmp<ExecutableTaskInvocation> {
public:
  ExecutableTaskInvocation() = delete;
  ExecutableTaskInvocation(task_id_t const &task_id, TaskBinding const &binding);

public:
  task_id_t task_id;
  ExecutableTaskBinding binding;
};

struct TaskReturnAccessor { 
  template <typename T>
  TypedFuture<T> get_returned_future();

  template <typename T>
  TypedFutureMap<T> get_returned_future_map();
};


TaskReturnAccessor execute_task(LegionConfig const &config, 
                                TaskInvocation const &,
                                RuntimeBacking const &backing);


}

#endif
