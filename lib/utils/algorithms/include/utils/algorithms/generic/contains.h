#ifndef _FLEXFLOW_LIB_UTILS_ALGORITHMS_INCLUDE_UTILS_ALGORITHMS_CONTAINS_H
#define _FLEXFLOW_LIB_UTILS_ALGORITHMS_INCLUDE_UTILS_ALGORITHMS_CONTAINS_H

#include "find.h"

namespace FlexFlow {

template <typename Container>
bool contains(Container const &c, typename Container::value_type const &e) {
  return find<Container>(c, e) != c.cend();
}

template <typename C>
bool contains_key(C const &m, typename C::key_type const &k) {
  return m.find(k) != m.end();
}


} // namespace FlexFlow

#endif
