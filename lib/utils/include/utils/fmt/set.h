#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_FMT_SET_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_FMT_SET_H

#include "utils/check_fmtable.h"
#include "utils/containers/sorted.h"
#include "utils/join_strings.h"
#include <doctest/doctest.h>
#include <fmt/format.h>
#include <set>
#include <vector>

namespace fmt {

template <typename T, typename Char>
struct formatter<::std::set<T>,
                 Char,
                 std::enable_if_t<!detail::has_format_as<std::set<T>>::value>>
    : formatter<::std::string> {
  template <typename FormatContext>
  auto format(::std::set<T> const &m, FormatContext &ctx) const
      -> decltype(ctx.out()) {
    CHECK_FMTABLE(T);

    std::vector<T> items = ::FlexFlow::sorted(m);
    std::string result = ::FlexFlow::join_strings(
        items.cbegin(), items.cend(), ", ", [](T const &t) {
          return fmt::to_string(t);
        });
    return formatter<std::string>::format("{" + result + "}", ctx);
  }
};

} // namespace fmt

namespace FlexFlow {

template <typename T>
std::ostream &operator<<(std::ostream &s, std::set<T> const &x) {
  CHECK_FMTABLE(T);

  return s << fmt::to_string(x);
}

} // namespace FlexFlow

namespace doctest {

template <typename T>
struct StringMaker<std::set<T>> {
  static String convert(std::set<T> const &m) {
    return toString(fmt::to_string(m));
  }
};

} // namespace doctest

#endif
