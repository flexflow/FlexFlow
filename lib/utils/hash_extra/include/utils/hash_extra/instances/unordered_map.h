#ifndef _FLEXFLOW_LIB_UTILS_HASH_EXTRA_INCLUDE_UTILS_HASH_EXTRA_STD_UNORDERED_MAP_H
#define _FLEXFLOW_LIB_UTILS_HASH_EXTRA_INCLUDE_UTILS_HASH_EXTRA_STD_UNORDERED_MAP_H

#include <unordered_map>

namespace std {

template <typename K, typename V>
struct hash<std::unordered_map<K, V>> {
  size_t operator()(std::unordered_map<K, V> const &m) const {
    return get_std_hash(items(m));
  }
};


} // namespace FlexFlow

#endif
