#ifndef _FLEXFLOW_LIB_UTILS_HASH_EXTRA_INCLUDE_UTILS_HASH_EXTRA_GET_STD_HASH_H
#define _FLEXFLOW_LIB_UTILS_HASH_EXTRA_INCLUDE_UTILS_HASH_EXTRA_GET_STD_HASH_H

#include <cstddef>
#include <functional>

namespace FlexFlow {

template <class T>
std::size_t get_std_hash(T const &v) {
  std::hash<T> hasher;
  return hasher(v);
}

} // namespace FlexFlow

#endif
