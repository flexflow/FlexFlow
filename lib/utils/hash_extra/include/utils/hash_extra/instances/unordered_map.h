#ifndef _FLEXFLOW_LIB_UTILS_HASH_EXTRA_INCLUDE_UTILS_HASH_EXTRA_STD_UNORDERED_MAP_H
#define _FLEXFLOW_LIB_UTILS_HASH_EXTRA_INCLUDE_UTILS_HASH_EXTRA_STD_UNORDERED_MAP_H

#include "utils/hash_extra/get_std_hash.h"
#include "utils/hash_extra/instances/pair.h"
#include "utils/hash_extra/instances/unordered_set.h"
#include <functional>
#include <unordered_map>
#include <unordered_set>

namespace std {

template <typename K, typename V>
struct hash<std::unordered_map<K, V>> {
  size_t operator()(std::unordered_map<K, V> const &m) const {
    using ::FlexFlow::get_std_hash;

    std::unordered_set<std::pair<K, V>> items = {m.begin(), m.end()};
    return get_std_hash(items);
  }
};

} // namespace std

#endif
