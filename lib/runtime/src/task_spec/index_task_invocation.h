#ifndef _FLEXFLOW_RUNTIME_SRC_INDEX_TASK_INVOCATION_H
#define _FLEXFLOW_RUNTIME_SRC_INDEX_TASK_INVOCATION_H

#include "task_invocation.h"
#include "typed_task_invocation.h"

namespace FlexFlow {

using IndexTaskArgSpec = variant<ConcreteArgSpec,
                                 IndexArgSpec,
                                 CheckedTypedFuture,
                                 CheckedTypedFutureMap,
                                 ArgRefSpec,
                                 TaskInvocationSpec>;

template <typename T>
using IndexTypedTaskArg =
    variant<IndexArg<T>, TypedFutureMap<T>, TypedIndexTaskInvocation<T>>;

struct IndexTaskBinding {
public:
  IndexTaskBinding() = delete;
  IndexTaskBinding(parallel_tensor_guid_t const &);
  IndexTaskBinding(slot_id const &);
  IndexTaskBinding(MachineView const &);

  void bind(slot_id, parallel_tensor_guid_t const &);
  void bind(slot_id, ParallelTensorSpec const &);

  template <typename T>
  void bind_arg(slot_id name, StandardTypedTaskArg<T> const &arg) {
    this->standard_binding.bind_arg(name, arg);
  }

  template <typename T>
  void bind_arg(slot_id name, TypedFutureMap<T> const &);

  template <typename T>
  void bind_arg(slot_id name, TypedIndexTaskInvocation<T> const &);

  template <typename F,
            typename T = decltype(std::declval<F>()(
                std::declval<Legion::DomainPoint>()))>
  void bind_index_arg(slot_id name, F const &f) {
    this->insert_arg_spec(name, IndexArgSpec::create(f));
  }

public:
  void insert_arg_spec(slot_id, IndexTaskArgSpec const &);

  TaskBinding standard_binding;
};

struct IndexTaskInvocation : public use_visitable_cmp<IndexTaskInvocation> {
public:
  IndexTaskInvocation() = delete;
  IndexTaskInvocation(task_id_t const &task_id, IndexTaskBinding const &binding)
      : task_id(task_id), binding(binding) {}

public:
  task_id_t task_id;
  IndexTaskBinding binding;
};

} // namespace FlexFlow

VISITABLE_STRUCT(::FlexFlow::IndexTaskInvocation, task_id, binding);

#endif
