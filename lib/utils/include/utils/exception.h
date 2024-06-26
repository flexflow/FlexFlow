#ifndef _FLEXFLOW_UTILS_INCLUDE_EXCEPTION_H
#define _FLEXFLOW_UTILS_INCLUDE_EXCEPTION_H

#include "utils/exception.decl.h"
#include "utils/fmt.h"
#include <stdexcept>
#include <tl/expected.hpp>

namespace FlexFlow {

template <typename T, typename E>
T throw_if_unexpected(tl::expected<T, E> const &r) {
  if (r.has_value()) {
    return r.value();
  } else {
    throw std::runtime_error(fmt::to_string(r.error()));
  }
}

template <typename... T>
std::runtime_error mk_runtime_error(fmt::format_string<T...> fmt_str,
                                    T &&...args) {
  return std::runtime_error(
      fmt::vformat(fmt_str, fmt::make_format_args(args...)));
}

} // namespace FlexFlow

#endif
