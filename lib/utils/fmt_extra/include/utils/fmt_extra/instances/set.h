#ifndef _FLEXFLOW_LIB_UTILS_FMT_EXTRA_INCLUDE_UTILS_FMT_EXTRA_INSTANCES_SET_H
#define _FLEXFLOW_LIB_UTILS_FMT_EXTRA_INCLUDE_UTILS_FMT_EXTRA_INSTANCES_SET_H

#include <set>
#include <fmt/format.h>
#include "utils/string_extra/surrounded.h"
#include "utils/string_extra/join_strings.h"
#include "utils/fmt_extra/is_fmtable.h"

namespace fmt {

template <typename T>
struct formatter<::std::set<T>,
                 ::std::enable_if_t<::FlexFlow::is_fmtable_v<T>, char>>
    : formatter<::std::string> {
  auto format(::std::set<T> const &m, format_context &ctx) const
      -> decltype(ctx.out()) {
    using namespace ::FlexFlow;

    std::string result =
        surrounded('{', '}', join_strings(m, ", ", [](T const &t) {
                     return element_to_string(t);
                   }));
    return formatter<std::string>::format(result, ctx);
  }
};

} // namespace FlexFlow

#endif
