#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_SUM_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_SUM_H

namespace FlexFlow {

/**
 * @details An empty container vacuously has sum 0
 **/
template <typename Container, typename Element = typename Container::value_type>
Element sum(Container const &container) {
  Element result = 0;
  for (Element const &element : container) {
    result += element;
  }
  return result;
}

template <typename Container,
          typename ConditionF,
          typename Element = typename Container::value_type>
Element sum_where(Container const &container, ConditionF const &condition) {
  Element result = 0;
  for (Element const &element : container) {
    if (condition(element)) {
      result += element;
    }
  }
  return result;
}

} // namespace FlexFlow

#endif
