#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_FMT_PAIR_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_FMT_PAIR_H

#include "utils/check_fmtable.h"
#include <fmt/format.h>
#include <utility>

namespace fmt {

template <typename L, typename R, typename Char>
struct formatter<
    ::std::pair<L, R>,
    Char,
    std::enable_if_t<!detail::has_format_as<::std::pair<L, R>>::value>>
    : formatter<::std::string> {
  template <typename FormatContext>
  auto format(::std::pair<L, R> const &m, FormatContext &ctx)
      -> decltype(ctx.out()) {
    CHECK_FMTABLE(L);
    CHECK_FMTABLE(R);

    std::string result = fmt::format("{{{}, {}}}", m.first, m.second);

    return formatter<std::string>::format(result, ctx);
  }
};

} // namespace fmt

namespace FlexFlow {

template <typename L, typename R>
std::ostream &operator<<(std::ostream &s, std::pair<L, R> const &m) {
  CHECK_FMTABLE(L);
  CHECK_FMTABLE(R);

  return s << fmt::to_string(m);
}

} // namespace FlexFlow

#endif
