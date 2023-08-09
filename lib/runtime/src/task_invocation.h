#ifndef _FLEXFLOW_RUNTIME_SRC_TASK_SPEC_H
#define _FLEXFLOW_RUNTIME_SRC_TASK_SPEC_H

#include "arg_ref.h"
#include "concrete_arg.h"
#include "index_arg.h"
#include "kernels/ff_handle.h"
#include "parallel_tensor_guid_t.h"
#include "parallel_tensor_spec.h"
#include "pcg/machine_view.h"
#include "profiling.h"
#include "serialization.h"
#include "task_signature.h"
#include "tasks.h"
#include "typed_future.h"
#include "typed_future_map.h"
#include "utils/type_index.h"
#include "utils/variant.h"
#include "utils/visitable.h"

namespace FlexFlow {

enum class ArgSlotType { INDEX, STANDARD };

template <typename T>
struct TypedTaskInvocation;
template <typename T>
struct TypedIndexTaskInvocation;
struct TaskInvocationSpec;

using StandardArgSpec = variant<ConcreteArgSpec,
                                CheckedTypedFuture,
                                CheckedTypedFutureMap,
                                ArgRefSpec,
                                TaskInvocationSpec>;

template <typename T>
using TypedTaskArg = variant<T,
                             IndexArg<T>,
                             TypedFuture<T>,
                             TypedFutureMap<T>,
                             ArgRef<T>,
                             TypedTaskInvocation<T>,
                             TypedIndexTaskInvocation<T>>;

template <typename T>
using StandardTypedTaskArg =
    variant<T, TypedFuture<T>, ArgRef<T>, TypedTaskInvocation<T>>;

template <typename T>
using IndexTypedTaskArg =
    variant<IndexArg<T>, TypedFutureMap<T>, TypedIndexTaskInvocation<T>>;

std::type_index get_type_index(ArgSpec);

template <typename T>
TaskInvocationSpec create_task_invocation_spec(TypedTaskInvocation<T> const &);

struct TaskBinding {
public:
  static TaskBinding index_launch(parallel_tensor_guid_t const &);
  static TaskBinding index_launch(slot_id const &);
  static TaskBinding index_launch(MachineView const &);
  static TaskBinding standard_launch();
  static TaskBinding sync_type_dependent_launch(parallel_tensor_guid_t);
  static TaskBinding sync_type_dependent_launch(slot_id);

  void bind(slot_id, parallel_tensor_guid_t const &);
  void bind(slot_id, ParallelTensorSpec const &);

  template <typename T>
  void bind_arg(slot_id name, ArgRef<T> const &a) {
    this->insert_arg_spec(name, ArgRefSpec::create(a));
  }

  template <typename T>
  void bind_arg(slot_id name, TypedTaskArg<T> const &);

  template <typename T>
  void bind_arg(slot_id name, StandardTypedTaskArg<T> const &);

  template <typename T>
  void bind_arg(slot_id name, TypedTaskInvocation<T> const &invoc) {
    this->insert_arg_spec(name, create_task_invocation_spec(invoc));
  }

  template <typename T>
  void bind_arg(slot_id name, TypedIndexTaskInvocation<T> const &invoc);

  template <typename T>
  void bind_arg(slot_id name, T const &t) {
    this->insert_arg_spec(name, ConcreteArgSpec::create(t));
  }

  template <typename T>
  void bind_arg(slot_id name, TypedFuture<T> const &f) {
    this->insert_arg_spec(name, CheckedTypedFuture::create(f));
  }

  template <typename T>
  void bind_arg(slot_id name, TypedFutureMap<T> const &fm) {
    this->insert_arg_spec(name, CheckedTypedFutureMap::create(fm));
  }

private:
  void insert_arg_spec(slot_id name, ArgSpec const &arg_spec);

private:
  std::unordered_map<slot_id, ArgSpec> arg_bindings;
  std::unordered_map<slot_id, parallel_tensor_guid_t> bindings;
};

struct TaskInvocation : public use_visitable_cmp<TaskInvocation> {
public:
  TaskInvocation() = delete;
  TaskInvocation(task_id_t const &task_id, TaskBinding const &binding)
      : task_id(task_id), binding(binding) {}

public:
  task_id_t task_id;
  TaskBinding binding;
};

/* TaskArgumentFormat compile_task_invocation(TaskInvocation const &); */

/* std::unordered_map<Legion::DomainPoint, TaskArgumentFormat>
 * compile_index_task_invocation(TaskSignature const &signature, */
/*                                                                                           TaskBinding
 * const &binding); */

} // namespace FlexFlow

VISITABLE_STRUCT(::FlexFlow::TaskInvocation, task_id, binding);

#endif
