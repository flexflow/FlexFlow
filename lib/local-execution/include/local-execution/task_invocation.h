#ifndef _FLEXFLOW_LOCAL_EXECUTION_TASK_INVOCATION_H
#define _FLEXFLOW_LOCAL_EXECUTION_TASK_INVOCATION_H

#include "local-execution/slot_grad_id.dtg.h"
#include "local-execution/slot_id_t.dtg.h"
#include "local-execution/task_arg_spec.dtg.h"
#include "local-execution/task_id_t.dtg.h"
#include "local-execution/task_signature.dtg.h"
#include "local-execution/tensor_guid_spec.dtg.h"

namespace FlexFlow {

struct TaskBinding {
  TaskBinding() = default;

  void bind(int, TensorGuidSpec const &);
  void bind(slot_id_t, TensorGuidSpec const &);

  template <typename T>
  void bind_arg(int name, T const &t) {
    this->bind_arg<T>(slot_id_t{name}, t);
  }

  template <typename T>
  void bind_arg(slot_id_t name, T const &t) {
    this->insert_arg_spec(name, TaskArgSpec{ConcreteArgSpec::create(t)});
  }

  template <typename T>
  void bind_arg(int name, RuntimeArgRef<T> const &t) {
    this->bind_arg<T>(slot_id_t{name}, t);
  }

  template <typename T>
  void bind_arg(slot_id_t name, RuntimeArgRef<T> const &ref) {
    this->insert_arg_spec(name, TaskArgSpec{RuntimeArgRefSpec::create(ref)});
  }

  bool operator==(TaskBinding const &other) const;
  bool operator!=(TaskBinding const &other) const;

  std::unordered_map<SlotGradId, TensorGuidSpec> const &
      get_tensor_bindings() const;
  std::unordered_map<slot_id_t, TaskArgSpec> const &get_arg_bindings() const;

private:
  std::unordered_map<SlotGradId, TensorGuidSpec> tensor_bindings;
  std::unordered_map<slot_id_t, TaskArgSpec> arg_bindings;

private:
  void insert_arg_spec(slot_id_t name, TaskArgSpec const &arg_spec);
  std::tuple<decltype(tensor_bindings) const &, decltype(arg_bindings) const &>
      tie() const;
};

struct TaskInvocation {
public:
  TaskInvocation() = delete;
  TaskInvocation(task_id_t task_id, TaskBinding const &binding)
      : task_id(task_id), binding(binding) {}

public:
  task_id_t task_id;
  TaskBinding binding;
};

bool is_invocation_valid(TaskSignature const &sig, TaskInvocation const &inv);

} // namespace FlexFlow

#endif
