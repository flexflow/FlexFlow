#ifndef _FLEXFLOW_RUNTIME_INCLUDE_RUNTIME_TASK_SPEC_TASK_RETURN_ACCESSOR_H
#define _FLEXFLOW_RUNTIME_INCLUDE_RUNTIME_TASK_SPEC_TASK_RETURN_ACCESSOR_H

#include "legion.h"
#include "utils/optional.h"
#include "utils/type_index.h"
#include "utils/variant.h"

namespace FlexFlow {

template <typename T>
struct TypedFuture;
template <typename T>
struct TypedFutureMap;
struct CheckedTypedFuture;
struct CheckedTypedFutureMap;

struct TaskReturnAccessor {
  TaskReturnAccessor(optional<std::type_index>, Legion::Future const &);
  TaskReturnAccessor(optional<std::type_index>, Legion::FutureMap const &);

  void wait() const;

  template <typename T>
  TypedFuture<T> get_returned_future() const;

  template <typename T>
  TypedFutureMap<T> get_returned_future_map() const;

  CheckedTypedFuture get_returned_future() const;
  CheckedTypedFutureMap get_returned_future_map() const;

  variant<Legion::Future, Legion::FutureMap> get_future_unsafe() const;
  optional<std::type_index> get_type_index() const;
};

} // namespace FlexFlow

#endif
