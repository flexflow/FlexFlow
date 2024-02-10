#ifndef _FLEXFLOW_HASH_UTILS_H
#define _FLEXFLOW_HASH_UTILS_H

#include "containers.h"
#include "hash-utils-core.h"

namespace std {
template <typename T>
struct hash<std::unordered_set<T>> {
  size_t operator()(std::unordered_set<T> const &s) const {
    auto sorted = sorted_by(s, ::FlexFlow::compare_by<T>([](T const &t) {
                              return get_std_hash(t);
                            }));
    return get_std_hash(sorted);
  }
};

template <typename K, typename V>
struct hash<std::unordered_map<K, V>> {
  size_t operator()(std::unordered_map<K, V> const &m) const {
    return get_std_hash(items(m));
  }
};

} // namespace std

#endif
