#ifndef _FLEXFLOW_UTILS_INCLUDE_EXPECTED_H
#define _FLEXFLOW_UTILS_INCLUDE_EXPECTED_H

#include <tl/expected.hpp>
#include "utils/fmt.h"
#include <string>
#include <optional>

namespace FlexFlow {

template <typename... Args>
tl::unexpected<std::string> error_msg(Args &&...args) {
  return tl::make_unexpected(fmt::format(std::forward<Args>(args)...));
}

template <typename T, typename E>
std::optional<T> optional_from_expected(tl::expected<T, E> const &x) {
  if (x.has_value()) {
    return x.value();
  } else {
    return std::nullopt;
  }
}

} // namespace FlexFlow

#endif
