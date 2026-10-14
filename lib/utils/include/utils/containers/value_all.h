#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_VALUE_ALL_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_VALUE_ALL_H

#include "utils/containers/transform.h"
#include "utils/optional.h"

namespace FlexFlow {

template <typename T>
std::vector<T> value_all(std::vector<std::optional<T>> const &v) {
  return transform(v, [&](std::optional<T> const &element) {
    return unwrap(element, [&] {
      PANIC(fmt::format(
          "value_all expected all elements to have values, but received {}",
          v));
    });
  });
}

template <typename T>
std::set<T> value_all(std::set<std::optional<T>> const &v) {
  return transform(v, [&](std::optional<T> const &element) {
    return unwrap(element, [&] {
      PANIC(fmt::format(
          "value_all expected all elements to have values, but received {}",
          v));
    });
  });
}

} // namespace FlexFlow

#endif
