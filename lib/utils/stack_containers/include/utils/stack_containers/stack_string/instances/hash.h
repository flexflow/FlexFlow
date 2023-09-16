#ifndef _FLEXFLOW_LIB_UTILS_STACK_CONTAINERS_INCLUDE_UTILS_STACK_CONTAINERS_STACK_STRING_HASH_H
#define _FLEXFLOW_LIB_UTILS_STACK_CONTAINERS_INCLUDE_UTILS_STACK_CONTAINERS_STACK_STRING_HASH_H

#include <cstddef>

namespace std {

template <typename Char, std::size_t MAXSIZE>
struct hash<::FlexFlow::stack_basic_string<Char, MAXSIZE>> {
  size_t
      operator()(::FlexFlow::stack_basic_string<Char, MAXSIZE> const &s) const {
    return get_std_hash(s.contents);
  }
};

} // namespace std

#endif
