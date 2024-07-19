#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_HASH_PAIR_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_HASH_PAIR_H

#include "utils/hash-utils.h"
#include <utility>

namespace std {

template <typename L, typename R>
struct hash<std::pair<L, R>> {
  size_t operator()(std::pair<L, R> const &p) const {
    size_t seed = 283746;

    ::FlexFlow::hash_combine(seed, p.first);
    ::FlexFlow::hash_combine(seed, p.second);

    return seed;
  }
};

} // namespace std

#endif
