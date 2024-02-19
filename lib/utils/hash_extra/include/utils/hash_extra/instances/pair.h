#ifndef _FLEXFLOW_LIB_UTILS_HASH_EXTRA_INCLUDE_UTILS_HASH_EXTRA_STD_PAIR_H
#define _FLEXFLOW_LIB_UTILS_HASH_EXTRA_INCLUDE_UTILS_HASH_EXTRA_STD_PAIR_H

#include <utility>
#include <functional>
#include "utils/hash_extra/hash_combine.h"

namespace std {

template <typename L, typename R>
struct hash<std::pair<L, R>> {
  size_t operator()(std::pair<L, R> const &p) const {
    size_t seed = 283746;

    hash_combine(seed, p.first);
    hash_combine(seed, p.second);

    return seed;
  }
};

} // namespace FlexFlow

#endif
