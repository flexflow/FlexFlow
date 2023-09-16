#ifndef _FLEXFLOW_LIB_UTILS_HASH_EXTRA_INCLUDE_UTILS_HASH_EXTRA_STD_VECTOR_H
#define _FLEXFLOW_LIB_UTILS_HASH_EXTRA_INCLUDE_UTILS_HASH_EXTRA_STD_VECTOR_H

namespace FlexFlow {

template <typename T>
struct hash<std::vector<T>> {
  size_t operator()(std::vector<T> const &vec) const {
    size_t seed = 0;
    iter_hash(seed, vec.cbegin(), vec.cend());
    return seed;
  }
};


} // namespace FlexFlow

#endif
