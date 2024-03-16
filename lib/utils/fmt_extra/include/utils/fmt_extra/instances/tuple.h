#ifndef _FLEXFLOW_LIB_UTILS_FMT_EXTRA_INCLUDE_UTILS_FMT_EXTRA_INSTANCES_TUPLE_H
#define _FLEXFLOW_LIB_UTILS_FMT_EXTRA_INCLUDE_UTILS_FMT_EXTRA_INSTANCES_TUPLE_H

#include <tuple>
#include <fmt/format.h>
#include "utils/string_extra/surrounded.h"
#include "utils/string_extra/join_strings.h"
#include "utils/fmt_extra/all_are_fmtable.h"
#include "utils/fmt_extra/element_to_string.h"
#include "range/v3/view/transform.hpp"

namespace fmt {

template <typename... Ts>
struct formatter<::std::tuple<Ts...>,
                 ::std::enable_if_t<::FlexFlow::all_are_fmtable_v<Ts...>, char>>
    : formatter<::std::string> {
  auto format(::std::tuple<Ts...> const &m, format_context &ctx) const
    -> decltype(ctx.out()) {
    using namespace ::FlexFlow;

    std::vector<std::string> v = to_vector(
        transform(m, [](auto const &t) { return element_to_string(t); }));
    std::string result = surrounded(
        '<', '>', join_strings(v, ", ", [](std::string const &s) { return s; }));
    return formatter<std::string>::format(result, ctx);
  }
};

} // namespace FlexFlow

#endif
