#ifndef _FLEXFLOW_LIB_UTILS_FMT_EXTRA_INCLUDE_UTILS_FMT_EXTRA_INSTANCES_MAP_H
#define _FLEXFLOW_LIB_UTILS_FMT_EXTRA_INCLUDE_UTILS_FMT_EXTRA_INSTANCES_MAP_H

#include <map>
#include <fmt/format.h>
#include "utils/fmt_extra/all_are_fmtable.h"
#include "utils/algorithms/sorting.h"
#include "utils/fmt_extra/element_to_string.h"
#include "utils/string_extra/join_strings.h"
#include "utils/string_extra/surrounded.h"

namespace fmt {

template <typename K, typename V>
struct formatter<::std::map<K, V>,
                 ::std::enable_if_t<::FlexFlow::all_are_fmtable_v<K, V>, char>>
    : formatter<::std::string> {
  auto format(::std::map<K, V> const &m, format_context &ctx) const 
    -> decltype(ctx.out()) {
    using namespace ::FlexFlow;

    std::string result = surrounded(
        '{', '}', join_strings(sorted(m), ", ", [](std::pair<K, V> const &kv) {
          return fmt::format("{}: {}",
                             element_to_string(kv.first),
                             element_to_string(kv.second));
        }));
    return formatter<std::string>::format(result, ctx);
  }
};

} // namespace FlexFlow

#endif
