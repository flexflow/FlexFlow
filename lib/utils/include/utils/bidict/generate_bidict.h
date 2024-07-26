#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_BIDICT_GENERATE_BIDICT_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_BIDICT_GENERATE_BIDICT_H

#include "utils/bidict/bidict.h"
#include "utils/containers/get_element_type.h"
#include "utils/containers/transform.h"
#include <type_traits>

namespace FlexFlow {

template <typename F,
          typename C,
          typename K = get_element_type_t<C>,
          typename V = std::invoke_result_t<F, K>>
bidict<K, V> generate_bidict(C const &c, F const &f) {
  static_assert(is_hashable<K>::value,
                "Key type should be hashable (but is not)");
  static_assert(is_hashable<V>::value,
                "Value type should be hashable (but is not)");

  auto transformed = transform(c, [&](K const &k) -> std::pair<K, V> {
    return {k, f(k)};
  });
  return {transformed.cbegin(), transformed.cend()};
}

} // namespace FlexFlow

#endif
