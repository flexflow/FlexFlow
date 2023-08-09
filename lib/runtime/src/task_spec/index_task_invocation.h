#ifndef _FLEXFLOW_RUNTIME_INCLUDE_RUNTIME_TASK_SPEC_INDEX_TASK_INVOCATION_H
#define _FLEXFLOW_RUNTIME_INCLUDE_RUNTIME_TASK_SPEC_INDEX_TASK_INVOCATION_H

#include "arg_ref.h"
#include "parallel_tensor_spec.h"
#include "pcg/machine_view.h"
#include "pcg/parallel_tensor_guid_t.h"
#include "runtime/task_spec/concrete_arg.h"
#include "runtime/task_spec/index_arg.h"
#include "runtime/task_spec/slot_id.h"
#include "runtime/task_spec/typed_future.h"
#include "runtime/task_spec/typed_future_map.h"
#include "runtime/task_spec/typed_task_invocation.h"
#include "standard_task_invocation.h"
#include "tasks.h"

namespace FlexFlow {

using IndexTaskArgSpec = variant<ConcreteArgSpec,
                                 IndexArgSpec,
                                 CheckedTypedFuture,
                                 CheckedTypedFutureMap,
                                 RuntimeArgRefSpec,
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
  void bind_arg(slot_id name, T const &arg) {
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

  StandardTaskBinding standard_binding;
};
FF_VISITABLE_STRUCT_NONSTANDARD_CONSTRUCTION(IndexTaskBinding,
                                             standard_binding);

struct IndexTaskInvocation {
  req<task_id_t> task_id;
  req<IndexTaskBinding> binding;
};
FF_VISITABLE_STRUCT(IndexTaskInvocation, task_id, binding);

} // namespace FlexFlow

#endif
