#ifndef _FLEXFLOW_LIB_UTILS_TUPLE_EXTRA_INCLUDE_UTILS_TUPLE_EXTRA_FOR_EACH_H
#define _FLEXFLOW_LIB_UTILS_TUPLE_EXTRA_INCLUDE_UTILS_TUPLE_EXTRA_FOR_EACH_H

#include "utils/tuple_extra/transform.h"

namespace FlexFlow {

template <typename F, typename... Ts>
void for_each(std::tuple<Ts...> const &tup, F f) {
  transform(tup, [&f](auto const &x) {
    f(x);
    return std::tuple{};
  });
}

} // namespace FlexFlow

#endif
