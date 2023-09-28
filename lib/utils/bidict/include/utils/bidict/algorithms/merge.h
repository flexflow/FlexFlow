#ifndef _FLEXFLOW_LIB_UTILS_BIDICT_INCLUDE_UTILS_BIDICT_ALGORITHMS_MERGE_H
#define _FLEXFLOW_LIB_UTILS_BIDICT_INCLUDE_UTILS_BIDICT_ALGORITHMS_MERGE_H

namespace FlexFlow {

template <typename K, typename V>
bidict<K, V> merge_maps(bidict<K, V> const &lhs, bidict<K, V> const &rhs) {
  assert(are_disjoint(keys(lhs), keys(rhs)));

  bidict<K, V> result;
  for (auto const &kv : lhs) {
    result.equate(kv.first, kv.second);
  }
  for (auto const &kv : rhs) {
    result.equate(kv.first, kv.second);
  }

  return result;
}

} // namespace FlexFlow

#endif
