#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_FMT_OPTIONAL_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_FMT_OPTIONAL_H

#include "utils/check_fmtable.h"
#include <fmt/format.h>
#include <optional>

namespace fmt {

template <typename T, typename Char>
struct formatter<
    ::std::optional<T>,
    Char,
    std::enable_if_t<!detail::has_format_as<::std::optional<T>>::value>>
    : formatter<::std::string> {
  template <typename FormatContext>
  auto format(::std::optional<T> const &m, FormatContext &ctx) const
      -> decltype(ctx.out()) {
    CHECK_FMTABLE(T);

    std::string result;
    if (m.has_value()) {
      result = fmt::to_string(m.value());
    } else {
      result = "nullopt";
    }

    return formatter<std::string>::format(result, ctx);
  }
};

template <typename Char>
struct formatter<std::nullopt_t, Char> : formatter<std::string> {
  template <typename FormatContext>
  auto format(std::nullopt_t, FormatContext &ctx) const -> decltype(ctx.out()) {
    return formatter<std::string>::format("nullopt", ctx);
  }
};

} // namespace fmt

namespace FlexFlow {

template <typename T>
std::ostream &operator<<(std::ostream &s, std::optional<T> const &t) {
  CHECK_FMTABLE(T);

  return s << fmt::to_string(t);
}

inline std::ostream &operator<<(std::ostream &s, std::nullopt_t) {
  return s << "nullopt";
}

} // namespace FlexFlow

#endif
