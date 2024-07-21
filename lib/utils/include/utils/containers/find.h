#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_FIND_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_FIND_H

#include <algorithm>

namespace FlexFlow {

template <typename Container>
typename Container::const_iterator
    find(Container const &c, typename Container::value_type const &e) {
  return std::find(c.cbegin(), c.cend(), e);
}

} // namespace FlexFlow

#endif
