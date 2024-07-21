#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_CONTAINS_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_CONTAINS_H

#include "utils/containers/find.h"

namespace FlexFlow {

template <typename Container>
bool contains(Container const &c, typename Container::value_type const &e) {
  return find<Container>(c, e) != c.cend();
}

} // namespace FlexFlow

#endif
