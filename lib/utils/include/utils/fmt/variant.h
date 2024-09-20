#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_FMT_VARIANT_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_FMT_VARIANT_H

#include <fmt/format.h>
#include <variant>

namespace fmt {

template <typename... Ts, typename Char>
struct formatter<std::variant<Ts...>, Char>
    /* std::enable_if_t<!detail::has_format_as<::tl::expected<T, E>>::value>> */
    : formatter<::std::string> {
  template <typename FormatContext>
  auto format(std::variant<Ts...> const &m, FormatContext &ctx)
      -> decltype(ctx.out()) {

    std::string result =
        std::visit([&](auto &&x) { return fmt::to_string(x); }, m);

    return formatter<std::string>::format(result, ctx);
  }
};

} // namespace fmt

namespace FlexFlow {

template <typename... Ts>
std::ostream &operator<<(std::ostream &s, std::variant<Ts...> const &v) {
  return s << fmt::to_string(v);
}

} // namespace FlexFlow

#endif
