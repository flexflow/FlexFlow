#ifndef _FLEXFLOW_LIB_UTILS_FMT_EXTRA_INCLUDE_UTILS_FMT_EXTRA_INSTANCES_PAIR_H
#define _FLEXFLOW_LIB_UTILS_FMT_EXTRA_INCLUDE_UTILS_FMT_EXTRA_INSTANCES_PAIR_H

#include "utils/fmt_extra/all_are_fmtable.h"
#include "utils/fmt_extra/element_to_string.h"
#include <fmt/format.h>
#include <utility>

namespace fmt {

template <typename T1, typename T2>
struct formatter<
    ::std::pair<T1, T2>,
    ::std::enable_if_t<::FlexFlow::all_are_fmtable_v<T1, T2>, char>>
    : formatter<::std::string> {
  auto format(::std::pair<T1, T2> const &m, format_context &ctx) const
      -> decltype(ctx.out()) {
    using namespace ::FlexFlow;

    std::string result = fmt::format(
        "<{}, {}>", element_to_string(m.first), element_to_string(m.second));
    return formatter<std::string>::format(result, ctx);
  }
};

} // namespace fmt

#endif
