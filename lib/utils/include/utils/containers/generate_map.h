#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_GENERATE_MAP_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_GENERATE_MAP_H

#include "utils/containers/get_element_type.h"
#include "utils/containers/vector_of.h"
#include "utils/containers/vector_transform.h"
#include "utils/type_traits_core.h"
#include <unordered_map>

namespace FlexFlow {

template <typename F,
          typename C,
          typename K = get_element_type_t<C>,
          typename V = std::invoke_result_t<F, K>>
std::unordered_map<K, V> generate_map(C const &c, F const &f) {
  static_assert(is_hashable_v<K>, "Key type should be hashable (but is not)");

  auto transformed =
      vector_transform(vector_of(c), [&](K const &k) -> std::pair<K, V> {
        return {k, f(k)};
      });
  return {transformed.cbegin(), transformed.cend()};
}

} // namespace FlexFlow

#endif
