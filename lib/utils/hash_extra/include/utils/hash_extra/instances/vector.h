#ifndef _FLEXFLOW_LIB_UTILS_HASH_EXTRA_INCLUDE_UTILS_HASH_EXTRA_STD_VECTOR_H
#define _FLEXFLOW_LIB_UTILS_HASH_EXTRA_INCLUDE_UTILS_HASH_EXTRA_STD_VECTOR_H

#include <vector>
#include <functional>
#include "utils/hash_extra/iter_hash.h"

namespace std {

template <typename T>
struct hash<std::vector<T>> {
  size_t operator()(std::vector<T> const &vec) const {
    using ::FlexFlow::iter_hash;

    size_t seed = 0;
    iter_hash(seed, vec.cbegin(), vec.cend());
    return seed;
  }
};


} // namespace std

#endif
