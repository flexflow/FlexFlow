#ifndef _FLEXFLOW_LIB_UTILS_FMT_EXTRA_INCLUDE_UTILS_FMT_EXTRA_INSTANCES_UNORDERED_SET_H
#define _FLEXFLOW_LIB_UTILS_FMT_EXTRA_INCLUDE_UTILS_FMT_EXTRA_INSTANCES_UNORDERED_SET_H

#include "utils/fmt_extra/element_to_string.h"
#include "utils/fmt_extra/is_fmtable.h"
#include "utils/string_extra/join_strings.h"
#include "utils/string_extra/surrounded.h"
#include <algorithm>
#include <fmt/format.h>
#include <unordered_set>

namespace fmt {

template <typename T>
struct formatter<::std::unordered_set<T>,
                 ::std::enable_if_t<::FlexFlow::is_fmtable_v<T>, char>>
    : formatter<::std::string> {
  auto format(::std::unordered_set<T> const &m, format_context &ctx) const
      -> decltype(ctx.out()) {
    using namespace ::FlexFlow;

    std::vector<T> sorted{m.begin(), m.end()};
    std::sort(sorted.begin(), sorted.end());
    std::string result =
        surrounded('{', '}', join_strings(sorted, ", ", [](T const &t) {
                     return element_to_string(t);
                   }));
    return formatter<std::string>::format(result, ctx);
  }
};

} // namespace fmt

#endif
