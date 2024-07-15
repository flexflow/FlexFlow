#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_FMT_UNORDERED_SET_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_FMT_UNORDERED_SET_H

#include "utils/check_fmtable.h"
#include "utils/join_strings.h"
#include <fmt/format.h>
#include <unordered_set>
#include "utils/containers/sorted.h"

namespace fmt {

template <typename T, typename Char>
struct formatter<
    ::std::unordered_set<T>,
    Char,
    std::enable_if_t<!detail::has_format_as<std::unordered_set<T>>::value>>
    : formatter<::std::string> {
  template <typename FormatContext>
  auto format(::std::unordered_set<T> const &m, FormatContext &ctx)
      -> decltype(ctx.out()) {
    CHECK_FMTABLE(T);

    std::vector<T> in_order = ::FlexFlow::sorted(m);
    std::string result =
        ::FlexFlow::join_strings(in_order.cbegin(), in_order.cend(), ", ", [](T const &t) {
          return fmt::to_string(t);
        });
    return formatter<std::string>::format("{" + result + "}", ctx);
  }
};

} // namespace fmt

namespace FlexFlow {

template <typename T>
std::ostream &operator<<(std::ostream &s, std::unordered_set<T> const &x) {
  CHECK_FMTABLE(T);

  return s << fmt::to_string(x);
}

} // namespace FlexFlow

#endif
