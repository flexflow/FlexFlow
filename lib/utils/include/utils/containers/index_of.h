#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_INDEX_OF_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_INDEX_OF_H

#include <algorithm>
#include <optional>

namespace FlexFlow {

/**
 * @details If multiple `e` are present within the container, the function
 * returns the index of the first appearance
 **/
template <typename Container, typename Element>
std::optional<std::size_t> index_of(Container const &c, Element const &e) {
  auto it = std::find(c.cbegin(), c.cend(), e);
  if (it == c.cend()) {
    return std::nullopt;
  } else {
    return std::distance(c.cbegin(), it);
  }
}

} // namespace FlexFlow

#endif
