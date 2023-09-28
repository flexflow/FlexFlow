#ifndef _FLEXFLOW_LIB_UTILS_BIDICT_INCLUDE_UTILS_BIDICT_ALGORITHMS_TRANSFORM_H
#define _FLEXFLOW_LIB_UTILS_BIDICT_INCLUDE_UTILS_BIDICT_ALGORITHMS_TRANSFORM_H

namespace FlexFlow {

template <typename K, typename V, typename F, typename K2>
bidict<K2, V> transform_keys(bidict<K, V> const &m, F const &f) {
  bidict<K2, V> result;
  for (auto const &kv : m) {
    result.equate(f(kv.first), kv.second);
  }
  return result;
}

template <typename K, typename V, typename F, typename V2>
bidict<K, V2> transform_values(bidict<K, V> const &m, F const &f) {
  bidict<K, V2> result;
  for (auto const &kv : m) {
    result.equate({kv.first, f(kv.second)});
  }
  return result;
}


} // namespace FlexFlow

#endif
