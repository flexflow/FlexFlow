#ifndef _FLEXFLOW_LIB_UTILS_TUPLE_EXTRA_INCLUDE_UTILS_TUPLE_EXTRA_PREPEND_H
#define _FLEXFLOW_LIB_UTILS_TUPLE_EXTRA_INCLUDE_UTILS_TUPLE_EXTRA_PREPEND_H

#include <tuple>

namespace FlexFlow {

template <typename Head, typename... Tail>
std::tuple<Head, Tail...> tuple_prepend(Head const &h,
                                        std::tuple<Tail...> const &tail) {
  return std::tuple_cat(std::tuple<Head>{h}, tail);
}

} // namespace FlexFlow

#endif
