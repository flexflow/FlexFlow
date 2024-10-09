#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_FMT_EXPECTED_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_FMT_EXPECTED_H

#include "fmt/format.h"
#include "utils/check_fmtable.h"
#include <doctest/doctest.h>
#include <tl/expected.hpp>
#include <utility>

namespace fmt {

template <typename T, typename E, typename Char>
struct formatter<
    ::tl::expected<T, E>,
    Char,
    std::enable_if_t<!detail::has_format_as<::tl::expected<T, E>>::value>>
    : formatter<::std::string> {
  template <typename FormatContext>
  auto format(::tl::expected<T, E> const &m, FormatContext &ctx) const
      -> decltype(ctx.out()) {

    std::string result;
    if (m.has_value()) {
      result = fmt::format("expected({})", m.value());
    } else {
      result = fmt::format("unexpected({})", m.error());
    }

    return formatter<std::string>::format(result, ctx);
  }
};

} // namespace fmt

namespace FlexFlow {

template <typename T, typename E>
std::ostream &operator<<(std::ostream &s, tl::expected<T, E> const &t) {
  CHECK_FMTABLE(T);
  CHECK_FMTABLE(E);

  return s << fmt::to_string(t);
}

} // namespace FlexFlow

namespace doctest {

template <typename T, typename E>
struct StringMaker<tl::expected<T, E>> {
  static String convert(tl::expected<T, E> const &m) {
    return toString(fmt::to_string(m));
  }
};

} // namespace doctest

#endif
