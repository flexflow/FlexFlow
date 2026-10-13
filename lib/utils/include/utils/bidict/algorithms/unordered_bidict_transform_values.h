#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_BIDICT_ALGORITHMS_UNORDERED_BIDICT_TRANSFORM_VALUES_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_BIDICT_ALGORITHMS_UNORDERED_BIDICT_TRANSFORM_VALUES_H

#include "utils/bidict/unordered_bidict.h"

namespace FlexFlow {

template <typename K,
          typename V,
          typename F,
          typename V2 = std::invoke_result_t<F, V>>
unordered_bidict<K, V2>
    unordered_bidict_transform_values(unordered_bidict<K, V> const &m, F &&f) {
  unordered_bidict<K, V2> result;
  for (auto const &kv : m) {
    result.equate_strict({kv.first, f(kv.second)});
  }
  return result;
}

} // namespace FlexFlow

#endif
