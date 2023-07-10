#ifndef _FLEXFLOW_RUNTIME_SRC_TASK_SPEC_H
#define _FLEXFLOW_RUNTIME_SRC_TASK_SPEC_H

#include "arg_ref.h"
#include "runtime/task_spec/concrete_arg.h"
#include "runtime/task_spec/index_arg.h"
#include "kernels/ff_handle.h"
#include "pcg/parallel_tensor_guid_t.h"
#include "parallel_tensor_spec.h"
#include "pcg/machine_view.h"
#include "profiling.h"
#include "serialization.h"
#include "task_signature.h"
#include "tasks.h"
#include "runtime/task_spec/typed_future.h"
#include "runtime/task_spec/typed_future_map.h"
#include "runtime/task_spec/typed_task_invocation.h"
#include "utils/type_index.h"
#include "utils/variant.h"
#include "utils/visitable.h"

namespace FlexFlow {

/** 
 * \class ArgSlotType
 * \brief Enum class denoting the two types of ArgSlots
 * 
 * An enum class denoting the two types of argument slots:
 *  (1) INDEX,
 *  (2) STANDARD;
*/
enum class ArgSlotType { INDEX, STANDARD };

template <typename T>
struct TypedTaskInvocation;
template <typename T>
struct TypedIndexTaskInvocation;

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

/**
 * \fn std::type_index get_type_index(StandardArgSpec)
 * \param standardargspec StandardArgSpec
 * \brief Gets type index from StandardArgSpec
*/
std::type_index get_type_index(StandardArgSpec standardargspec);

/**
 * \fn TaskInvocationSpec create_task_invocation_spec(TypedTaskInvocation<T> const &)
 * \param typedtaskinvocation TypedTaskInvocation
 * \brief Creates TaskInvocationSpec<T> given TypedTaskInvocation<T>
*/
template <typename T>
TaskInvocationSpec create_task_invocation_spec(TypedTaskInvocation<T> const &typedtaskinvocation);

/**
 * \class TaskBinding
 * \brief describes binding methods that insert argument specification
 * 
 * Has binding methods that insert argument specifications; 
 * Has properties: arg_bindings (std::unordered_map<...>)and bindings (std::unordered_map<...>);
*/
struct TaskBinding {
public:
  static TaskBinding standard_launch();
  static TaskBinding sync_type_dependent_launch(parallel_tensor_guid_t);
  static TaskBinding sync_type_dependent_launch(slot_id);

  /**
   * \fn void bind(slot_id name, parallel_tensor_guid_t const &ptgs)
   * \param name slot_id of argument specification
   * \param ptgs parallel_tensor_guid_t
  */
  void bind(slot_id name, parallel_tensor_guid_t const &ptgs);

  /**
   * \fn void bind(slot_id name, ParallelTensorSpec const &)
   * \param name slot_id of argument specification
   * \param pts ParallelTensorSpec
  */
  void bind(slot_id name, ParallelTensorSpec const &pts);

  /**
   * \fn void bind_arg(slot_id name, ArgRef<T> const &a)
   * \param name slot_id of argument specification
   * \param a ArgRef to create ArgRefSpec from
   * \brief Insert name and ArgRefSpec (created from a) as argument specifications
  */
  template <typename T>
  void bind_arg(slot_id name, ArgRef<T> const &a) {
    this->insert_arg_spec(name, ArgRefSpec::create(a));
  }

  /**
   * \fn void bind_arg(slot_id name, TypedTaskArg<T> const &a)
   * \param name slot_id of argument specification
   * \param a TypedTaskArg
   * \brief todo
  */
  template <typename T>
  void bind_arg(slot_id name, TypedTaskArg<T> const &a);

  /**
   * \fn void bind_arg(slot_id name, StandardTypedTaskArg<T> const &a)
   * \param name slot_id of argument specification
   * \param a StandardTypedTaskArg
   * \brief todo
  */
  template <typename T>
  void bind_arg(slot_id name, StandardTypedTaskArg<T> const &a);

  /**
   * \fn void bind_arg(slot_id name, TypedTaskInvocation<T> const &invoc)
   * \param name slot_id of argument specification
   * \param invoc TypedTaskInvocation used to create task invocation specification
   * \brief Inserts name (slot_id) and a task invocation specification (created from invoc) as argument specifications
  */
  template <typename T>
  void bind_arg(slot_id name, TypedTaskInvocation<T> const &invoc) {
    this->insert_arg_spec(name, create_task_invocation_spec(invoc));
  }

  /**
   * \fn void bind_arg(slot_id name, TypedIndexTaskInvocation<T> const &invoc)
   * \param name slot_id of argument specification
   * \param invoc TypedIndexTaskInvocation
   * \brief todo
  */
  template <typename T>
  void bind_arg(slot_id name, TypedIndexTaskInvocation<T> const &invoc);

  /**
   * \fn void bind_arg(slot_id name, T const &t)
   * \param name slot_id of argument specification
   * \param t T used to create ConcreteArgSpec object
   * \brief Inserts name and ConcreteArgSpec (created from t) as argument specifications
  */
  template <typename T>
  void bind_arg(slot_id name, T const &t) {
    this->insert_arg_spec(name, ConcreteArgSpec::create(t));
  }

  /**
   * \fn void bind_arg(slot_id name, TypedFuture<T> const &f)
   * \param name slot_id of argument specification
   * \param f TypedFuture used to create CheckedTypedFuture object
   * \brief Inserts name and CheckedTypedFuture (created from f) as argument specifications
  */
  template <typename T>
  void bind_arg(slot_id name, TypedFuture<T> const &f) {
    this->insert_arg_spec(name, CheckedTypedFuture::create(f));
  }

  /**
   * \fn void bind_arg(slot_id name, StandardTypedTaskArg<T> const &a)
   * \param name slot_id of argument specification
   * \param a StandardTypedTaskArg
   * \brief todo
  */
  template <typename T>
  void bind_arg(slot_id name, TypedFutureMap<T> const &fm) {
    this->insert_arg_spec(name, CheckedTypedFutureMap::create(fm));
  }

private:
  /**
   * \fn void insert_arg_spec(slot_id name, StandardArgSpec const &arg_spec)
   * \param name slot_id of argument specification
   * \param arg_spec StandardArgSpec to inserted as argument specification
   * \brief Inserts a name a argument specification to arg_bindings
  */
  void insert_arg_spec(slot_id name, StandardArgSpec const &arg_spec);

private:
  std::unordered_map<slot_id, StandardArgSpec> arg_bindings;
  std::unordered_map<slot_id, parallel_tensor_guid_t> bindings;
};

/**
 * \class TaskInvocation
 * \brief holds task_id and its binding
 * 
 * Deleted default constructor; Must pass in a task_id_t and a TaskBinding to create object); 
 * Compiles down to ExecutableTaskInvocation;
*/
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
