#ifndef _FLEXFLOW_RUNTIME_SRC_TYPED_TASK_INVOCATION_H
#define _FLEXFLOW_RUNTIME_SRC_TYPED_TASK_INVOCATION_H

#include "task_invocation.h"

namespace FlexFlow {

template <typename T> TypedTaskInvocation<T> ensure_return_type(TaskInvocation const &);

template <typename T> 
struct TypedTaskInvocation { 
public:
  TypedTaskInvocation() = delete;
  
  friend TypedTaskInvocation ensure_return_type<T>(TaskInvocation const &);

  operator TaskInvocation() const;
private:
  TypedTaskInvocation(TaskInvocation const &);

  TaskInvocation invocation;
};

template <typename T> TypedIndexTaskInvocation<T> ensure_index_return_type(TaskInvocation const &);

template <typename T>
struct TypedIndexTaskInvocation {
  TypedIndexTaskInvocation() = delete;

  friend TypedIndexTaskInvocation ensure_index_return_type<T>(TaskInvocation const &);

  operator TaskInvocation() const;
private:
  TypedIndexTaskInvocation(TaskInvocation const &);

  TaskInvocation invocation;
};

template <typename T>
TypedTaskInvocation<T> ensure_return_type(TaskInvocation const &invocation) { 
  optional<std::type_index> signature_return_type = get_signature(invocation.task_id).get_return_type();
  std::type_index asserted_return_type = type_index<T>();
  if (!signature_return_type.has_value()) {
    throw mk_runtime_error("Task {} has no return type (asserted type {})",
                           asserted_return_type);
  }
  if (signature_return_type.value() != asserted_return_type) {
    throw mk_runtime_error("Task {} does not have asserted return type (asserted type {}, signature type {})",
                           get_name(invocation.task_id),
                           asserted_return_type,
                           signature_return_type.value()
                           );
  }

  return TypedTaskInvocation<T>(invocation);
}

template <typename T>
TypedIndexTaskInvocation<T> ensure_index_return_type(TaskInvocation const &invocation);


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
  TaskInvocationSpec(std::type_index const &type_idx, TaskInvocation const &invocation) 
    : type_idx(type_idx), invocation(invocation)
  { }
  
  std::type_index type_idx;
  TaskInvocation invocation;
};

template <typename T> 
TaskInvocationSpec create_task_invocation_spec(TypedTaskInvocation<T> const &invoc) {
  return TaskInvocationSpec::create<T>(invoc);
}


}

#endif
