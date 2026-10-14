#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_OPTIONAL_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_OPTIONAL_H

#include "utils/fmt/optional.h"
#include <libassert/assert.hpp>
#include <rapidcheck.h>

namespace FlexFlow {

template <typename T, typename F, typename U = std::invoke_result_t<F, T>>
U and_then(std::optional<T> const &o, F &&f) {
  if (o.has_value()) {
    return f(o.value());
  } else {
    return std::nullopt;
  }
}

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
    PANIC("Failure in unwrap");
  }
}

template <typename T>
T const &assert_unwrap(std::optional<T> const &o) {
  ASSERT(o.has_value());
  return o.value();
}

template <typename T>
T expect(std::optional<T> const &x, std::string const &err) {
  ASSERT(x.has_value(), err);
  return x.value();
}

} // namespace FlexFlow

#endif
