#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_FLATMAP_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_FLATMAP_H

#include "utils/containers/extend.h"
#include "utils/containers/get_element_type.h"
#include <type_traits>

namespace FlexFlow {

template <typename In,
          typename F,
          typename Out = typename std::invoke_result_t<F, In>::value_type>
std::vector<Out> flatmap(std::vector<In> const &v, F const &f) {
  std::vector<Out> result;
  for (auto const &elem : v) {
    extend(result, f(elem));
  }
  return result;
}

template <typename In,
          typename F,
          typename Out = get_element_type_t<std::invoke_result_t<F, In>>>
std::unordered_set<Out> flatmap(std::unordered_set<In> const &v, F const &f) {
  std::unordered_set<Out> result;
  for (auto const &elem : v) {
    extend(result, f(elem));
  }
  return result;
}

template <typename Out, typename In>
std::unordered_set<Out> flatmap_v2(std::unordered_set<In> const &v,
                                   std::unordered_set<Out> (*f)(In const &)) {
  std::unordered_set<Out> result;
  for (auto const &elem : v) {
    extend(result, f(elem));
  }
  return result;
}

} // namespace FlexFlow

#endif
