#ifndef _FLEXFLOW_RUNTIME_SRC_TASK_SPEC_H
#define _FLEXFLOW_RUNTIME_SRC_TASK_SPEC_H

#include "arg_ref.h"
#include "concrete_arg.h"
#include "index_arg.h"
#include "kernels/ff_handle.h"
#include "parallel_tensor_guid_t.h"
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

enum class InvocationType { INDEX, STANDARD };

enum class ArgSlotType { INDEX, STANDARD };

enum class IsGrad { YES, NO };

struct ParallelTensorSpec : public use_visitable_cmp<ParallelTensorSpec> {
public:
  ParallelTensorSpec() = delete;
  ParallelTensorSpec(parallel_tensor_guid_t, IsGrad is_grad = IsGrad::NO);

  ParallelTensorSpec grad() const;

public:
  parallel_tensor_guid_t parallel_tensor_guid;
  IsGrad is_grad;
};

ParallelTensorSpec grad(parallel_tensor_guid_t const &);

template <typename T>
struct TypedTaskInvocation;
struct TaskInvocationSpec;

using ArgSpec = variant<ConcreteArgSpec,
                        IndexArgSpec,
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
                             TypedTaskInvocation<T>>;

std::type_index get_type_index(ArgSpec);

using IndexLaunchDomainSpec = variant<MachineView, parallel_tensor_guid_t>;

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
  void bind_arg(slot_id name, TypedTaskInvocation<T> const &invoc) {
    this->insert_arg_spec(name, create_task_invocation_spec(invoc));
  }

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

  template <typename F,
            typename T = decltype(std::declval<F>()(
                std::declval<Legion::DomainPoint>()))>
  void bind_index_arg(slot_id name, F const &f) {
    this->insert_arg_spec(name, IndexArgSpec::create(f));
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

template <typename T>
TypedTaskInvocation<T> ensure_return_type(TaskInvocation const &);

template <typename T>
struct TypedTaskInvocation {
public:
  TypedTaskInvocation() = delete;

  friend TypedTaskInvocation ensure_return_type<T>(TaskInvocation const &);

private:
  TypedTaskInvocation(TaskInvocation const &);

  TaskInvocation invocation;
};

template <typename T>
TypedTaskInvocation<T> ensure_return_type(TaskInvocation const &invocation) {
  optional<std::type_index> signature_return_type =
      get_signature(invocation.task_id).get_return_type();
  std::type_index asserted_return_type = type_index<T>();
  if (!signature_return_type.has_value()) {
    throw mk_runtime_error("Task {} has no return type (asserted type {})",
                           asserted_return_type);
  }
  if (signature_return_type.value() != asserted_return_type) {
    throw mk_runtime_error("Task {} does not have asserted return type "
                           "(asserted type {}, signature type {})",
                           get_name(invocation.task_id),
                           asserted_return_type,
                           signature_return_type.value());
  }

  return TypedTaskInvocation<T>(invocation);
}

struct TaskInvocationSpec {
  TaskInvocationSpec() = delete;

  TaskInvocation get_invocation() const {
    return this->invocation;
  }

  template <typename T>
  static TaskInvocationSpec create(TypedTaskInvocation<T> const &invocation) {
    return TaskInvocationSpec(type_index<T>(), invocation.invocation);
  }

private:
  TaskInvocationSpec(std::type_index const &type_idx,
                     TaskInvocation const &invocation)
      : type_idx(type_idx), invocation(invocation) {}

  std::type_index type_idx;
  TaskInvocation invocation;
};

template <typename T>
TaskInvocationSpec
    create_task_invocation_spec(TypedTaskInvocation<T> const &invoc) {
  return TaskInvocationSpec::create<T>(invoc);
}

/* TaskArgumentFormat compile_task_invocation(TaskInvocation const &); */

/* std::unordered_map<Legion::DomainPoint, TaskArgumentFormat>
 * compile_index_task_invocation(TaskSignature const &signature, */
/*                                                                                           TaskBinding
 * const &binding); */

} // namespace FlexFlow

VISITABLE_STRUCT(::FlexFlow::TaskInvocation, task_id, binding);

#endif
