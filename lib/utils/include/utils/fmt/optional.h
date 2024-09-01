#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_FMT_OPTIONAL_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_FMT_OPTIONAL_H

#include "utils/check_fmtable.h"
#include <fmt/format.h>
#include <optional>
#include <doctest/doctest.h>

namespace fmt {

template <typename T, typename Char>
struct formatter<
    ::std::optional<T>,
    Char,
    std::enable_if_t<!detail::has_format_as<::std::optional<T>>::value>>
    : formatter<::std::string> {
  template <typename FormatContext>
  auto format(::std::optional<T> const &m, FormatContext &ctx)
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

} // namespace fmt

namespace FlexFlow {

template <typename T>
std::ostream &operator<<(std::ostream &s, std::optional<T> const &t) {
  CHECK_FMTABLE(T);

  return s << fmt::to_string(t);
}

} // namespace FlexFlow

namespace doctest {

template <typename T>
struct StringMaker<std::optional<T>> {
  static String convert(std::optional<T> const &m) {
    return toString(fmt::to_string(m));
  }
};

} // namespace doctest


#endif
