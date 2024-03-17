#ifndef _FLEXFLOW_LIB_UTILS_TUPLE_EXTRA_INCLUDE_UTILS_TUPLE_EXTRA_TO_VECTOR_H
#define _FLEXFLOW_LIB_UTILS_TUPLE_EXTRA_INCLUDE_UTILS_TUPLE_EXTRA_TO_VECTOR_H

#include "utils/type_traits_extra/types_are_all_same.h"
#include "utils/tuple_extra/for_each.h"
#include <vector>
#include <type_traits>

namespace FlexFlow {

template <typename Head, typename... Rest>
std::enable_if_t<types_are_all_same_v<Head, Rest...>, std::vector<Head>>
    to_vector(std::tuple<Head, Rest...> const &tup) {
  std::vector<Head> result;
  for_each(tup, [&](Head const &h) { result.push_back(h); });
  return result;
}

} // namespace FlexFlow

#endif
