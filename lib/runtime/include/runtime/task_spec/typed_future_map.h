#ifndef _FLEXFLOW_RUNTIME_INCLUDE_RUNTIME_TASK_SPEC_TYPED_FUTURE_MAP_H
#define _FLEXFLOW_RUNTIME_INCLUDE_RUNTIME_TASK_SPEC_TYPED_FUTURE_MAP_H

#include "legion.h"
#include "utils/type_index.h"

namespace FlexFlow {

template <typename T>
struct TypedFutureMap {
  explicit TypedFutureMap(Legion::FutureMap const &future_map)
      : future_map(future_map) {}

  T get(Legion::DomainPoint const &p) {
    return future_map.get_result<T>(p);
  }

public:
  Legion::FutureMap future_map;
};

struct CheckedTypedFutureMap {
public:
  CheckedTypedFutureMap() = delete;

  template <typename T>
  TypedFutureMap<T> get() {
    assert(matches<T>(this->type));

    return {this->future_map};
  }

  template <typename T>
  static CheckedTypedFutureMap create(TypedFutureMap<T> const &fm) {
    return CheckedTypedFutureMap(type_index<T>(), fm.future_map);
  }

private:
  CheckedTypedFutureMap(std::type_index const &type_idx,
                        Legion::FutureMap const &future_map)
      : type(type_idx), future_map(future_map) {}

  friend struct TaskReturnAccessor;

  std::type_index type;
  Legion::FutureMap future_map;
};

} // namespace FlexFlow

#endif
