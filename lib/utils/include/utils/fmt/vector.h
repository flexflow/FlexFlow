#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_FMT_VECTOR_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_FMT_VECTOR_H

#include "utils/check_fmtable.h"
#include "utils/join_strings.h"
#include <fmt/format.h>
#include <vector>

namespace fmt {

template <typename T, typename Char>
struct formatter<
    ::std::vector<T>,
    Char,
    std::enable_if_t<!detail::has_format_as<std::vector<T>>::value>>
    : formatter<::std::string> {
  template <typename FormatContext>
  auto format(::std::vector<T> const &m, FormatContext &ctx) const
      -> decltype(ctx.out()) {
    CHECK_FMTABLE(T);

    std::string result =
        ::FlexFlow::join_strings(m.cbegin(), m.cend(), ", ", [](T const &t) {
          return fmt::to_string(t);
        });
    return formatter<std::string>::format("[" + result + "]", ctx);
  }
};

} // namespace fmt

namespace FlexFlow {

template <typename T>
std::ostream &operator<<(std::ostream &s, std::vector<T> const &v) {
  CHECK_FMTABLE(T);

  return s << fmt::to_string(v);
}

} // namespace FlexFlow

#endif
