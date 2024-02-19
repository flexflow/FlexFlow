#ifndef _FLEXFLOW_LIB_UTILS_FMT_EXTRA_INCLUDE_UTILS_FMT_EXTRA_INSTANCES_VARIANT_H
#define _FLEXFLOW_LIB_UTILS_FMT_EXTRA_INCLUDE_UTILS_FMT_EXTRA_INSTANCES_VARIANT_H

#include <variant>
#include <fmt/format.h>
#include "utils/fmt_extra/all_are_fmtable.h"

namespace fmt {

template <typename... Ts>
struct formatter<::std::variant<Ts...>,
                 ::std::enable_if_t<::FlexFlow::all_are_fmtable_v<Ts...>, char>>
    : formatter<::std::string> {
  auto format(::std::variant<Ts...> const &v, format_context &ctx) const
    -> decltype(ctx.out()) {
  return std::visit([](auto const &t) { return fmt::to_string(t); }, v);
  }
};

} // namespace fmt

#endif
