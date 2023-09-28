#ifndef _FLEXFLOW_LIB_UTILS_BIDICT_INCLUDE_UTILS_BIDICT_ALGORITHMS_GENERATE_H
#define _FLEXFLOW_LIB_UTILS_BIDICT_INCLUDE_UTILS_BIDICT_ALGORITHMS_GENERATE_H

namespace FlexFlow {

template <typename F, typename C, typename K, typename V>
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
