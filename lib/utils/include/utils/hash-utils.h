#ifndef _FLEXFLOW_HASH_UTILS_H
#define _FLEXFLOW_HASH_UTILS_H

#include "containers.h"
#include "hash-utils-core.h"
#include <set>
#include <map>

using namespace FlexFlow;

namespace std {
template <typename T>
struct hash<std::unordered_set<T>> {
  size_t operator()(std::unordered_set<T> const &s) const {
    size_t result = 0;
    unordered_container_hash(result, s);
    return result;
  }
};

template <typename T>
struct hash<std::set<T>> {
  size_t operator()(std::set<T> const &s) const {
    size_t result = 0;
    unordered_container_hash(result, s);
    return result;
  }
};

template <typename K, typename V>
struct hash<std::unordered_map<K, V>> {
  size_t operator()(std::unordered_map<K, V> const &m) const {
    size_t result = 0;
    unordered_container_hash(result, m);
    return result;
  }
};

template <typename K, typename V>
struct hash<std::map<K, V>> {
  size_t operator()(std::map<K, V> const &m) const {
    size_t result = 0;
    unordered_container_hash(result, m);
    return result;
  }
};


} // namespace std

#endif
