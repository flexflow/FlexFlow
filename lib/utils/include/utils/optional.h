#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_OPTIONAL_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_OPTIONAL_H

#include "utils/exception.h"
#include "utils/fmt/optional.h"
#include <rapidcheck.h>

namespace FlexFlow {

template <typename T, typename F>
T or_else(std::optional<T> const &o, F &&f) {
  if (o.has_value()) {
    return o.value();
  } else {
    return f();
  }
}

template <typename T, typename F>
T const &unwrap(std::optional<T> const &o, F const &f) {
  if (o.has_value()) {
    return o.value();
  } else {
    f();
    throw mk_runtime_error("Failure in unwrap");
  }
}

template <typename T>
T const &assert_unwrap(std::optional<T> const &o) {
  assert(o.has_value());
  return o.value();
}

} // namespace FlexFlow

#endif
