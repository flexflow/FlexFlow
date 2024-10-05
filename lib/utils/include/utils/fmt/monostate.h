#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_FMT_MONOSTATE_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_FMT_MONOSTATE_H

#include <fmt/format.h>
#include <variant>

namespace fmt {

template <typename Char>
struct formatter<
    ::std::monostate,
    Char,
    std::enable_if_t<!detail::has_format_as<::std::monostate>::value>>
    : formatter<::std::string> {
  template <typename FormatContext>
  auto format(::std::monostate const &, FormatContext &ctx)
      -> decltype(ctx.out()) {
    std::string result = "<monostate>";

    return formatter<std::string>::format(result, ctx);
  }
};

} // namespace fmt

namespace FlexFlow {

std::ostream &operator<<(std::ostream &, std::monostate const &);

} // namespace FlexFlow

#endif
