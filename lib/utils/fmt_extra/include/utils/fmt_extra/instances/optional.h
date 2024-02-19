#ifndef _FLEXFLOW_LIB_UTILS_FMT_EXTRA_INCLUDE_UTILS_FMT_EXTRA_INSTANCES_OPTIONAL_H
#define _FLEXFLOW_LIB_UTILS_FMT_EXTRA_INCLUDE_UTILS_FMT_EXTRA_INSTANCES_OPTIONAL_H

#include <optional>
#include <fmt/format.h>
#include "utils/fmt_extra/is_fmtable.h"

namespace fmt {

template <typename T>
struct formatter<::std::optional<T>,
                 ::std::enable_if_t<::FlexFlow::is_fmtable_v<T>, char>>
    : formatter<::std::string> {
  auto format(::std::optional<T> const &q, format_context &ctx) const
      -> decltype(ctx.out()) {
    std::string result;
    if (q.has_value()) {
      result = fmt::to_string(q.value());
    } else {
      result = "nullopt";
    }
    return formatter<std::string>::format(result, ctx);
  }
};

template <>
struct formatter<::std::nullopt_t, char>
    : formatter<::std::string> {
  auto format(::std::nullopt_t const &q, format_context &ctx) const
      -> decltype(ctx.out()) {
    return formatter<std::string>::format("nullopt", ctx);
  }
};

} // namespace FlexFlow

#endif
