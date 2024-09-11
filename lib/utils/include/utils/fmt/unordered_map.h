#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_FMT_UNORDERED_MAP_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_FMT_UNORDERED_MAP_H

#include "utils/check_fmtable.h"
#include "utils/fmt/pair.h"
#include "utils/join_strings.h"
#include <algorithm>
#include <doctest/doctest.h>
#include <fmt/format.h>
#include <unordered_map>
#include <vector>

namespace fmt {

template <typename K, typename V, typename Char>
struct formatter<
    ::std::unordered_map<K, V>,
    Char,
    std::enable_if_t<!detail::has_format_as<::std::unordered_map<K, V>>::value>>
    : formatter<::std::string> {
  template <typename FormatContext>
  auto format(::std::unordered_map<K, V> const &m, FormatContext &ctx)
      -> decltype(ctx.out()) {
    CHECK_FMTABLE(K);
    CHECK_FMTABLE(V);

    std::string result = ::FlexFlow::join_strings(
        m.cbegin(), m.cend(), ", ", [](std::pair<K, V> const &t) {
          return fmt::to_string(t);
        });
    // }

    return formatter<std::string>::format("{" + result + "}", ctx);
  }
};

} // namespace fmt

namespace FlexFlow {

template <typename K, typename V>
std::ostream &operator<<(std::ostream &s, std::unordered_map<K, V> const &m) {
  CHECK_FMTABLE(K);
  CHECK_FMTABLE(V);

  return s << fmt::to_string(m);
}

} // namespace FlexFlow

namespace doctest {

template <typename K, typename V>
struct StringMaker<std::unordered_map<K, V>> {
  static String convert(std::unordered_map<K, V> const &m) {
    return toString(fmt::to_string(m));
  }
};

} // namespace doctest

#endif
