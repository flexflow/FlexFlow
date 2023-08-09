#ifndef _FLEXFLOW_RUNTIME_SRC_TYPED_FUTURE_H
#define _FLEXFLOW_RUNTIME_SRC_TYPED_FUTURE_H

#include "legion.h"
#include "utils/type_index.h"

namespace FlexFlow {

template <typename T>
struct TypedFuture {
  explicit TypedFuture(Legion::Future const &future) : future(future) {}

  T get() {
    return future.get_result<T>();
  }

public:
  Legion::Future future;
};

struct CheckedTypedFuture {
public:
  CheckedTypedFuture() = delete;

  template <typename T>
  TypedFuture<T> get() const {
    assert(type_index<T>() == this->type_idx);

    return this->future;
  }

  Legion::Future get_unsafe() const {
    return this->future;
  }

  std::type_index get_type_idx() const {
    return this->type_idx;
  }

  template <typename T>
  static CheckedTypedFuture create(TypedFuture<T> const &f) {
    return CheckedTypedFuture(type_index<T>(), f.future);
  }

private:
  CheckedTypedFuture(std::type_index type_idx, Legion::Future const &future)
      : type_idx(type_idx), future(future) {}

  friend struct TaskReturnAccessor;

  std::type_index type_idx;
  Legion::Future future;
};

} // namespace FlexFlow

#endif
