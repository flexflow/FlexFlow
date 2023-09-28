#ifndef _FLEXFLOW_LIB_UTILS_STACK_CONTAINERS_INCLUDE_UTILS_STACK_CONTAINERS_STACK_VECTOR_HASH_H
#define _FLEXFLOW_LIB_UTILS_STACK_CONTAINERS_INCLUDE_UTILS_STACK_CONTAINERS_STACK_VECTOR_HASH_H

#include <functional>
#include "stack_vector.h"

namespace std {

template <typename T, std::size_t MAXSIZE>
struct hash<::FlexFlow::stack_vector<T, MAXSIZE>> {
  size_t operator()(::FlexFlow::stack_vector<T, MAXSIZE> const &v) {
    static_assert(::FlexFlow::is_hashable<T>::value,
                  "stack_vector elements must be hashable");
    size_t result = 0;
    iter_hash(result, v.cbegin(), v.cend());
    return result;
  }
};

} // namespace std

#endif
