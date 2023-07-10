#ifndef _FLEXFLOW_RUNTIME_INCLUDE_RUNTIME_TASK_SPEC_INDEX_TASK_INVOCATION_H
#define _FLEXFLOW_RUNTIME_INCLUDE_RUNTIME_TASK_SPEC_INDEX_TASK_INVOCATION_H

#include "parallel_tensor_spec.h"
#include "runtime/task_spec/typed_task_invocation.h"
#include "runtime/task_spec/concrete_arg.h"
#include "runtime/task_spec/index_arg.h"
#include "runtime/task_spec/typed_future.h"
#include "runtime/task_spec/typed_future_map.h"
#include "arg_ref.h"
#include "pcg/parallel_tensor_guid_t.h"
#include "pcg/machine_view.h"
#include "slot_id.h"
#include "task_invocation.h"

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

/**
 * \class IndexTaskBinding
 * \brief
 * 
 * Deleted default constructor; 
 * Create by passing in either: (1) parallel_tensor_guid_t, (2) slot_id, or (3) MachineView; 
 * 
*/
struct IndexTaskBinding {
public:
  IndexTaskBinding() = delete;
  IndexTaskBinding(parallel_tensor_guid_t const &);
  IndexTaskBinding(slot_id const &);
  IndexTaskBinding(MachineView const &);

  void bind(slot_id, parallel_tensor_guid_t const &);
  void bind(slot_id, ParallelTensorSpec const &);

  /**
   * \fn void bind_arg(slot_id name, StandardTypedTaskArg<T> const &arg)
   * \param name slot_id to be binded as arg to standard_binding
   * \param arg StandardTypeTypedTaskArg<T> to be binded as arg to standard_binding
   * \brief binds a slot_id and StandardTypedTaskArg to property: standard_binding (TaskBinding)
  */
  template <typename T>
  void bind_arg(slot_id name, StandardTypedTaskArg<T> const &arg) {
    this->standard_binding.bind_arg(name, arg);
  }

  /**
   * \fn void bind_arg(slot_id name, TypedFutureMap<T> const &)
   * \param name slot_id to be binded as arg to standard_binding
   * \param arg TypedFutureMap<T> to be binded as arg to standard_binding
   * 
   * todo: add brief (definition?)
  */
  template <typename T>
  void bind_arg(slot_id name, TypedFutureMap<T> const &arg);

  /**
   * \fn void bind_arg(slot_id name, TypedIndexTaskInvocation<T> const &arg)
   * \param name slot_id to be binded as arg to standard_binding
   * \param arg TypedIndexTaskInvocation<T> to be binded as arg to standard_binding
   * 
   * todo: add brief (definition?)
  */
  template <typename T>
  void bind_arg(slot_id name, TypedIndexTaskInvocation<T> const &arg);

  /**
   * \fn void bind_index_arg(slot_id name, F const &f)
   * \param name slot_id to be binded as arg to standard_binding
   * \param f F
   * \brief Creates an obj of IndexArgSpec (using f) and inserts name and the object as argument specifications
  */
  template <typename F,
            typename T = decltype(std::declval<F>()(
                std::declval<Legion::DomainPoint>()))>
  void bind_index_arg(slot_id name, F const &f) {
    this->insert_arg_spec(name, IndexArgSpec::create(f));
  }

public:
  /**
   * \fn void insert_arg_spec(slot_id, IndexTaskArgSpec const &)
   * \brief
   * \param name slot_id
   * \param arg_spec IndexTaskArgSpec passed into arg_bindings.insert()
   * 
   * todo add brief
  */
  void insert_arg_spec(slot_id name, IndexTaskArgSpec const &arg_spec);

  TaskBinding standard_binding;
};

/**
 * \class IndexTaskInvocation
 * \brief TaskInvocation with IndexTaskBinding and task_id_t
 * 
 * Compiled from OpTaskInvocation (different type of binding); 
 * Compiles down to ExecutableIndexTaskInvocation; 
 * Has task_id and binding (IndexTaskBinding); 
 * Deleted default constructor-must pass in 
 * task_id and binding;
*/
struct IndexTaskInvocation : public use_visitable_cmp<IndexTaskInvocation> {
public:
  IndexTaskInvocation() = delete;
  IndexTaskInvocation(task_id_t const &task_id, IndexTaskBinding const &binding)
      : task_id(task_id), binding(binding) {}

public:
  task_id_t task_id;
  IndexTaskBinding binding;
};

}

VISITABLE_STRUCT(::FlexFlow::IndexTaskInvocation, task_id, binding);

#endif
