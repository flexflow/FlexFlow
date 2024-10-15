#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_PRODUCT_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_PRODUCT_H

#include <numeric>

namespace FlexFlow {

/**
 * @details An empty container vacuously has product 1
 **/
template <typename Container, typename Element = typename Container::value_type>
Element product(Container const &container) {
  Element result = 1;
  for (Element const &element : container) {
    result *= element;
  }
  return result;
}

template <typename It>
typename It::value_type product(It begin, It end) {
  using Element = typename It::value_type;
  return std::accumulate(
      begin, end, 1, [](Element const &lhs, Element const &rhs) {
        return lhs * rhs;
      });
}

} // namespace FlexFlow

#endif
