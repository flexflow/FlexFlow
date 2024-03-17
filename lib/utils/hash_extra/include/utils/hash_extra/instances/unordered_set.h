#ifndef _FLEXFLOW_LIB_UTILS_HASH_EXTRA_INCLUDE_UTILS_HASH_EXTRA_STD_UNORDERED_SET_H
#define _FLEXFLOW_LIB_UTILS_HASH_EXTRA_INCLUDE_UTILS_HASH_EXTRA_STD_UNORDERED_SET_H

#include "utils/hash_extra/get_std_hash.h"
#include "utils/hash_extra/instances/vector.h"
#include <functional>
#include <unordered_set>

namespace std {

template <typename T>
struct hash<std::unordered_set<T>> {
  size_t operator()(std::unordered_set<T> const &s) const {
    using ::FlexFlow::get_std_hash;

    std::vector<T> sorted = {s.begin(), s.end()};
    std::sort(sorted.begin(), sorted.end(), [](T const &lhs, T const &rhs) {
      return get_std_hash(lhs) < get_std_hash(rhs);
    });
    return get_std_hash(sorted);
  }
};

} // namespace std

#endif
