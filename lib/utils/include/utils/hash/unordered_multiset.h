#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_HASH_UNORDERED_SET_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_HASH_UNORDERED_SET_H

#include "utils/hash-utils.h"
#include <unordered_set>

namespace std {

template <typename T>
struct hash<std::unordered_multiset<T>> {
  size_t operator()(std::unordered_multiset<T> const &s) const {
    size_t result = 0;
    ::FlexFlow::unordered_container_hash(result, s);
    return result;
  }
};

} // namespace std

#endif
