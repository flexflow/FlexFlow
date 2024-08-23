#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_FILTRANS_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_FILTRANS_H

#include "utils/type_traits_core.h"
#include <optional>
#include <vector>
#include <unordered_set>
#include <set>

namespace FlexFlow {

template <typename T>
struct unwrap_optional {
  static_assert("T is not a std::optional!");
};

template <typename T>
struct unwrap_optional<std::optional<T>> 
  : type_identity<T> {};

template <typename T>
using unwrap_optional_t = typename unwrap_optional<T>::type;

template <typename F, typename In, typename Out = unwrap_optional_t<std::invoke_result_t<F, In>>>
std::vector<Out> filtrans(std::vector<In> const &v, F f) {
  std::vector<Out> result;
  
  for (In const &i : v) {
    std::optional<Out> o = f(i);
    if (o.has_value()) {
      result.push_back(o.value());
    }
  }

  return result;
}

template <typename F, typename In, typename Out = unwrap_optional_t<std::invoke_result_t<F, In>>>
std::unordered_set<Out> filtrans(std::unordered_set<In> const &s, F f) {
  std::unordered_set<Out> result;
  
  for (In const &i : s) {
    std::optional<Out> o = f(i);
    if (o.has_value()) {
      result.insert(o.value());
    }
  }

  return result;
}

template <typename F, typename In, typename Out = unwrap_optional_t<std::invoke_result_t<F, In>>>
std::set<Out> filtrans(std::set<In> const &s, F f) {
  std::set<Out> result;
  
  for (In const &i : s) {
    std::optional<Out> o = f(i);
    if (o.has_value()) {
      result.insert(o.value());
    }
  }

  return result;
}


} // namespace FlexFlow

#endif
