#ifndef _FLEXFLOW_UTILS_INCLUDE_UTILS_OPTIONAL_H
#define _FLEXFLOW_UTILS_INCLUDE_UTILS_OPTIONAL_H

#include <optional>

namespace FlexFlow {

template <typename T, typename F>
T const &unwrap(std::optional<T> const &o, F f) {
  if (o.has_value()) {
    return o.value();
  } else {
    f();
    throw std::bad_optional_access{};
  }
}

} // namespace FlexFlow

#endif
