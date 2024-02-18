#ifndef _FLEXFLOW_LIB_UTILS_BIDICT_INCLUDE_UTILS_BIDICT_ALGORITHMS_GENERATE_H
#define _FLEXFLOW_LIB_UTILS_BIDICT_INCLUDE_UTILS_BIDICT_ALGORITHMS_GENERATE_H

#include "utils/bidict/bidict.h"
#include "utils/type_traits_extra/is_hashable.h"

namespace FlexFlow {

template <
  typename F, 
  typename C, 
  typename K = typename C::value_type, 
  typename V = std::invoke_result_t<F, K>
>
bidict<K, V> generate_bidict(C const &c, F const &f) {
  static_assert(is_hashable_v<K>,
                "Key type should be hashable (but is not)");
  static_assert(is_hashable_v<V>,
                "Value type should be hashable (but is not)");

  bidict<K, V> result;
  for (K const &k : c) {
    result.equate(k, f(k));
  }

  return result;
}

} // namespace FlexFlow

#endif
