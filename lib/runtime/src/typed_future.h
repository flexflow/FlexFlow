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

  ArgTypeRuntimeTag get_type_tag() const {
    return this->type_tag;
  }

  Legion::Future get_unsafe() const {
    return this->future;
  }

  template <typename T>
  static CheckedTypedFuture create(TypedFuture<T> const &f) {
    return CheckedTypedFuture(
        type_index<T>(), f.future, ArgTypeRuntimeTag::create<T>());
  }

private:
  CheckedTypedFuture(std::type_index const &type_idx,
                     Legion::Future const &future,
                     ArgTypeRuntimeTag const &type_tag)
      : future(future), type_tag(type_tag) {}
  friend struct TaskReturnAccessor;

  Legion::Future future;
  ArgTypeRuntimeTag type_tag;
};

} // namespace FlexFlow

#endif
