#ifndef _FLEXFLOW_UTILS_INCLUDE_EXCEPTION_H
#define _FLEXFLOW_UTILS_INCLUDE_EXCEPTION_H

#include "utils/fmt.h"
#include <stdexcept>
#include <tl/expected.hpp>
#include <fmt/format.h>

namespace FlexFlow {

#ifdef FF_REQUIRE_IMPLEMENTED
#define NOT_IMPLEMENTED()                                                      \
  static_assert(false,                                                         \
                "Function " __FUNC__ " not yet implemented " __FILE__          \
                ":" __LINE__);
#else
#define NOT_IMPLEMENTED()                                                      \
  throw not_implemented(__PRETTY_FUNCTION__, __FILE__, __LINE__);
#endif

class not_implemented : public std::logic_error {
public:
  not_implemented(std::string const &function_name,
                  std::string const &file_name,
                  int line);
};

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
