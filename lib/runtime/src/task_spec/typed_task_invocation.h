#ifndef _FLEXFLOW_RUNTIME_SRC_TYPED_TASK_INVOCATION_H
#define _FLEXFLOW_RUNTIME_SRC_TYPED_TASK_INVOCATION_H

#include "utils/type_index.h"
#include "utils/type_traits.h"
#include <memory>

namespace FlexFlow {

struct TaskInvocation;
struct IndexTaskInvocation;

template <typename T>
struct TypedTaskInvocation;
template <typename T>
struct TypedIndexTaskInvocation;

template <typename T>
TypedTaskInvocation<T> ensure_return_type(TaskInvocation const &);

template <typename T>
struct TypedTaskInvocation {
public:
  TypedTaskInvocation() = delete;

  friend TypedTaskInvocation ensure_return_type<T>(TaskInvocation const &);

  friend bool operator==(TypedTaskInvocation const &,
                         TypedTaskInvocation const &);
  friend bool operator!=(TypedTaskInvocation const &,
                         TypedTaskInvocation const &);
  friend bool operator<(TypedTaskInvocation const &,
                        TypedTaskInvocation const &);

  operator TaskInvocation() const;

private:
  TypedTaskInvocation(TaskInvocation const &);

  std::shared_ptr<TaskInvocation const> invocation;
};
static_assert(is_well_behaved_value_type<TypedTaskInvocation<int>>::value, "");

template <typename T>
TypedIndexTaskInvocation<T> ensure_return_type(IndexTaskInvocation const &);

template <typename T>
struct TypedIndexTaskInvocation {
  TypedIndexTaskInvocation() = delete;

  friend TypedIndexTaskInvocation
      ensure_return_type<T>(IndexTaskInvocation const &);

  friend bool operator==(TypedIndexTaskInvocation const &,
                         TypedIndexTaskInvocation const &);
  friend bool operator!=(TypedIndexTaskInvocation const &,
                         TypedIndexTaskInvocation const &);
  friend bool operator<(TypedIndexTaskInvocation const &,
                        TypedIndexTaskInvocation const &);

  operator TaskInvocation() const;

private:
  TypedIndexTaskInvocation(IndexTaskInvocation const &);

  std::shared_ptr<IndexTaskInvocation const> invocation;
};
static_assert(is_well_behaved_value_type<TypedIndexTaskInvocation<int>>::value,
              "");

struct TaskInvocationSpec {
  TaskInvocationSpec() = delete;

  TaskInvocation const &get_invocation() const {
    return *this->invocation;
  }

  template <typename T>
  static TaskInvocationSpec create(TypedTaskInvocation<T> const &invocation) {
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
static_assert(is_well_behaved_value_type<TaskInvocationSpec>::value, "");

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
static_assert(is_well_behaved_value_type<IndexTaskInvocationSpec>::value, "");

template <typename T>
TaskInvocationSpec
    create_task_invocation_spec(TypedTaskInvocation<T> const &invoc) {
  return TaskInvocationSpec::create<T>(invoc);
}

} // namespace FlexFlow

#endif
