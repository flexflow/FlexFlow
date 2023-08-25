#ifndef _FLEXFLOW_RUNTIME_INCLUDE_RUNTIME_TASK_SPEC_TYPED_TASK_INVOCATION_H
#define _FLEXFLOW_RUNTIME_INCLUDE_RUNTIME_TASK_SPEC_TYPED_TASK_INVOCATION_H

#include "utils/type_index.h"
#include "utils/type_traits.h"
#include <memory>

namespace FlexFlow {

struct TaskBinding;
struct TaskInvocation;
struct StandardTaskInvocation;
struct IndexTaskInvocation;
struct StandardTaskBinding;
struct IndexTaskBinding;

template <typename T>
struct TypedStandardTaskInvocation;
template <typename T>
struct TypedIndexTaskInvocation;

template <typename T>
TypedStandardTaskInvocation<T>
    ensure_return_type(StandardTaskInvocation const &);

template <typename T>
struct TypedStandardTaskInvocation {
public:
  TypedStandardTaskInvocation() = delete;

  friend TypedStandardTaskInvocation
      ensure_return_type<T>(StandardTaskInvocation const &);

  template <typename U>
  friend bool operator==(TypedStandardTaskInvocation<U> const &,
                         TypedStandardTaskInvocation<U> const &);

  template <typename U>
  friend bool operator!=(TypedStandardTaskInvocation<U> const &,
                         TypedStandardTaskInvocation<U> const &);

  template <typename U>
  friend bool operator<(TypedStandardTaskInvocation<U> const &,
                        TypedStandardTaskInvocation<U> const &);

  operator StandardTaskInvocation() const;

private:
  TypedStandardTaskInvocation(StandardTaskInvocation const &);

  std::shared_ptr<StandardTaskInvocation const> invocation;
};
CHECK_WELL_BEHAVED_VALUE_TYPE_NO_HASH(TypedStandardTaskInvocation<int>);

template <typename T>
TypedIndexTaskInvocation<T> ensure_return_type(IndexTaskInvocation const &);

template <typename T>
struct TypedIndexTaskInvocation {
  TypedIndexTaskInvocation() = delete;

  friend TypedIndexTaskInvocation
      ensure_return_type<T>(IndexTaskInvocation const &);

  template <typename U>
  friend bool operator==(TypedIndexTaskInvocation<U> const &,
                         TypedIndexTaskInvocation<U> const &);

  template <typename U>
  friend bool operator!=(TypedIndexTaskInvocation<U> const &,
                         TypedIndexTaskInvocation<U> const &);

  template <typename U>
  friend bool operator<(TypedIndexTaskInvocation<U> const &,
                        TypedIndexTaskInvocation<U> const &);

  operator TaskInvocation() const;

private:
  TypedIndexTaskInvocation(IndexTaskInvocation const &);

  std::shared_ptr<IndexTaskInvocation const> invocation;
};
CHECK_WELL_BEHAVED_VALUE_TYPE_NO_HASH(TypedIndexTaskInvocation<int>);

struct TaskInvocationSpec {
  TaskInvocationSpec() = delete;

  TaskInvocation const &get_invocation() const {
    return *this->invocation;
  }

  template <typename T>
  static TaskInvocationSpec
      create(TypedStandardTaskInvocation<T> const &invocation) {
    return TaskInvocationSpec(type_index<T>(), invocation.invocation);
  }

  friend bool operator==(TaskInvocationSpec const &,
                         TaskInvocationSpec const &);
  friend bool operator!=(TaskInvocationSpec const &,
                         TaskInvocationSpec const &);
  friend bool operator<(TaskInvocationSpec const &, TaskInvocationSpec const &);

private:
  TaskInvocationSpec(std::type_index const &, TaskInvocation const &);

  std::type_index type_idx;
  std::shared_ptr<TaskInvocation const> invocation;
};
CHECK_WELL_BEHAVED_VALUE_TYPE_NO_HASH(TaskInvocationSpec);

struct IndexTaskInvocationSpec {
  IndexTaskInvocationSpec() = delete;

  IndexTaskInvocation const &get_invocation() const {
    return *this->invocation;
  }

  template <typename T>
  static IndexTaskInvocationSpec
      create(TypedIndexTaskInvocation<T> const &invocation) {
    return IndexTaskInvocationSpec(type_index<T>(), invocation.invocation);
  }

  friend bool operator==(IndexTaskInvocationSpec const &,
                         IndexTaskInvocationSpec const &);
  friend bool operator!=(IndexTaskInvocationSpec const &,
                         IndexTaskInvocationSpec const &);
  friend bool operator<(IndexTaskInvocationSpec const &,
                        IndexTaskInvocationSpec const &);

private:
  IndexTaskInvocationSpec(std::type_index const &, TaskInvocation const &);

  std::type_index type_idx;
  std::shared_ptr<IndexTaskInvocation const> invocation;
};
CHECK_WELL_BEHAVED_VALUE_TYPE_NO_HASH(IndexTaskInvocationSpec);

template <typename T>
TaskInvocationSpec
    create_task_invocation_spec(TypedStandardTaskInvocation<T> const &invoc) {
  return TaskInvocationSpec::create<T>(invoc);
}

} // namespace FlexFlow

#endif
