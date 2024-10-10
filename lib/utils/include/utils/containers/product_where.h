#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_PRODUCT_WHERE_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_PRODUCT_WHERE_H

#include <numeric>

namespace FlexFlow {

/**
 * @details An empty container vacuously has product 1
 **/
template <typename Container,
          typename ConditionF,
          typename Element = typename Container::value_type>
Element product_where(Container const &container, ConditionF const &condition) {
  Element result = 1;
  for (Element const &element : container) {
    if (condition(element)) {
      result *= element;
    }
  }
  return result;
}

} // namespace FlexFlow

#endif
