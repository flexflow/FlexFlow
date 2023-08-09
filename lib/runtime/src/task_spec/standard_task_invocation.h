#ifndef _FLEXFLOW_RUNTIME_SRC_TASK_SPEC_STANDARD_TASK_INVOCATION_H
#define _FLEXFLOW_RUNTIME_SRC_TASK_SPEC_STANDARD_TASK_INVOCATION_H

#include "kernels/ff_handle.h"
#include "parallel_tensor_spec.h"
#include "pcg/machine_view.h"
#include "pcg/parallel_tensor_guid_t.h"
#include "runtime/profiling.h"
#include "runtime/task_spec/concrete_arg.h"
#include "runtime/task_spec/index_arg.h"
#include "runtime/task_spec/typed_future.h"
#include "runtime/task_spec/typed_future_map.h"
#include "runtime/task_spec/typed_task_invocation.h"
#include "runtime_arg_ref.h"
#include "serialization.h"
#include "task_signature.h"
#include "tasks.h"
#include "utils/type_index.h"
#include "utils/variant.h"
#include "utils/visitable.h"

namespace FlexFlow {

enum class ArgSlotType { INDEX, STANDARD };

template <typename T>
struct TypedStandardTaskInvocation;
template <typename T>
struct TypedIndexTaskInvocation;

using StandardArgSpec = variant<ConcreteArgSpec,
                                CheckedTypedFuture,
                                CheckedTypedFutureMap,
                                RuntimeArgRefSpec,
                                TaskInvocationSpec>;

template <typename T>
using TypedTaskArg = variant<T,
                             IndexArg<T>,
                             TypedFuture<T>,
                             TypedFutureMap<T>,
                             RuntimeArgRef<T>,
                             TypedStandardTaskInvocation<T>,
                             TypedIndexTaskInvocation<T>>;

template <typename T>
using StandardTypedTaskArg = variant<T,
                                     TypedFuture<T>,
                                     RuntimeArgRef<T>,
                                     TypedStandardTaskInvocation<T>>;

std::type_index get_type_index(StandardArgSpec);

template <typename T>
TaskInvocationSpec
    create_task_invocation_spec(TypedStandardTaskInvocation<T> const &);

struct StandardTaskBinding {
public:
  /* static TaskBinding standard_launch(); */
  /* static TaskBinding sync_type_dependent_launch(parallel_tensor_guid_t); */
  /* static TaskBinding sync_type_dependent_launch(slot_id); */

  void bind(slot_id, parallel_tensor_guid_t const &);
  void bind(slot_id, ParallelTensorSpec const &);

  template <typename T>
  void bind_arg(slot_id name, RuntimeArgRef<T> const &a) {
    this->insert_arg_spec(name, RuntimeArgRefSpec::create(a));
  }

  template <typename T>
  void bind_arg(slot_id name, TypedTaskArg<T> const &);

  template <typename T>
  void bind_arg(slot_id name, StandardTypedTaskArg<T> const &);

  template <typename T>
  void bind_arg(slot_id name, TypedStandardTaskInvocation<T> const &invoc) {
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
  void insert_arg_spec(slot_id name, StandardArgSpec const &arg_spec);

public:
  std::unordered_map<slot_id, StandardArgSpec> arg_bindings;
  std::unordered_map<slot_id, parallel_tensor_guid_t> bindings;
};
FF_VISITABLE_STRUCT_NONSTANDARD_CONSTRUCTION(StandardTaskBinding,
                                             arg_bindings,
                                             bindings);

struct StandardTaskInvocation {
  req<task_id_t> task_id;
  req<StandardTaskBinding> binding;
};
FF_VISITABLE_STRUCT(StandardTaskInvocation, task_id, binding);

/* TaskArgumentFormat compile_task_invocation(TaskInvocation const &); */

/* std::unordered_map<Legion::DomainPoint, TaskArgumentFormat>
 * compile_index_task_invocation(TaskSignature const &signature, */
/*                                                                                           TaskBinding
 * const &binding); */

} // namespace FlexFlow

#endif
