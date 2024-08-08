#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_HASH_VECTOR_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_HASH_VECTOR_H

#include "utils/hash-utils.h"
#include <vector>

namespace std {

template <typename T>
struct hash<std::vector<T>> {
  size_t operator()(std::vector<T> const &vec) const {
    size_t seed = 0;
    ::FlexFlow::iter_hash(seed, vec.cbegin(), vec.cend());
    return seed;
  }
};

} // namespace std

#endif
