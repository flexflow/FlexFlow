#ifndef _FLEXFLOW_RUNTIME_SRC_TASK_SPEC_TASK_INVOCATION_H
#define _FLEXFLOW_RUNTIME_SRC_TASK_SPEC_TASK_INVOCATION_H

#include "index_task_invocation.h"
#include "standard_task_invocation.h"
#include "utils/strong_typedef.h"
#include "utils/variant.h"

namespace FlexFlow {

struct TaskBinding {
  TaskBinding(IndexTaskBinding const &);
  TaskBinding(StandardTaskBinding const &);

  static TaskBinding sync_type_dependent_launch(slot_id);

  void bind(slot_id, parallel_tensor_guid_t const &);
  void bind(slot_id, ParallelTensorSpec const &);

  template <typename T>
  void bind_arg(slot_id, RuntimeArgRef<T> const &);

  template <typename T>
  void bind_arg(slot_id name, T const &t);

public:
  variant<StandardTaskBinding, IndexTaskBinding> binding;
};

struct TaskInvocation
    : strong_typedef<TaskInvocation,
                     variant<StandardTaskInvocation, IndexTaskInvocation>> {
  using strong_typedef::strong_typedef;

  TaskInvocation(slot_id, TaskBinding const &);
};

} // namespace FlexFlow

#endif
