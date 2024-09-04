#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_FMT_MULTISET_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_FMT_MULTISET_H

#include "utils/check_fmtable.h"
#include "utils/join_strings.h"
#include <doctest/doctest.h>
#include <fmt/format.h>
#include <set>

namespace fmt {

template <typename T, typename Char>
struct formatter<
    ::std::multiset<T>,
    Char,
    std::enable_if_t<!detail::has_format_as<std::multiset<T>>::value>>
    : formatter<::std::string> {
  template <typename FormatContext>
  auto format(::std::multiset<T> const &m, FormatContext &ctx)
      -> decltype(ctx.out()) {
    CHECK_FMTABLE(T);

    std::string result =
        ::FlexFlow::join_strings(m.cbegin(), m.cend(), ", ", [](T const &t) {
          return fmt::to_string(t);
        });
    // }
    return formatter<std::string>::format("{" + result + "}", ctx);
  }
};

} // namespace fmt

namespace FlexFlow {

template <typename T>
std::ostream &operator<<(std::ostream &s, std::multiset<T> const &x) {
  CHECK_FMTABLE(T);

  return s << fmt::to_string(x);
}

} // namespace FlexFlow

namespace doctest {

template <typename T>
struct StringMaker<std::multiset<T>> {
  static String convert(std::multiset<T> const &m) {
    return toString(fmt::to_string(m));
  }
};

} // namespace doctest

#endif
