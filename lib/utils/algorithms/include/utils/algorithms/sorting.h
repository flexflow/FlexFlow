#ifndef _FLEXFLOW_UTILS_INCLUDE_UTILS_ALGORITHMS_SORTING_H
#define _FLEXFLOW_UTILS_INCLUDE_UTILS_ALGORITHMS_SORTING_H

#include "utils/type_traits_extra/iterator.h"
#include <unordered_map>
#include "utils/backports/type_identity.h"
#include <algorithm>
#include <map>
#include <vector>
#include <functional>

namespace FlexFlow {

template <typename T, typename Enable = void>
struct sorted_elem_type {};

template <typename K, typename V>
struct sorted_elem_type<std::map<K, V>> : type_identity<std::pair<K, V>> {};

template <typename T>
using sorted_elem_type_t = typename sorted_elem_type<T>::type;

template <typename C, typename F>
std::vector<sorted_elem_type_t<C>> sorted_by(C const &c, F const &f) {
  using Elem = sorted_elem_type_t<C>;

  std::vector<Elem> result(c.begin(), c.end());
  inplace_sorted_by(result, f);
  return result;
}

template <typename C>
std::vector<sorted_elem_type_t<C>> sorted(C const &c) {
  using Elem = sorted_elem_type_t<C>;

  return sorted_by(c,
                   [](Elem const &lhs, Elem const &rhs) { return lhs < rhs; });
}

template <typename C, typename F, typename Elem>
void inplace_sorted_by(C &c, F const &f) {
  CHECK_SUPPORTS_ITERATOR_TAG(std::random_access_iterator_tag, C);

  auto custom_comparator = [&](Elem const &lhs, Elem const &rhs) -> bool {
    return f(lhs, rhs);
  };
  std::sort(c.begin(), c.end(), custom_comparator);
}

template <typename T>
struct sorted_elem_type<T> : type_identity<typename T::value_type> {};

template <typename K, typename V>
struct sorted_elem_type<std::unordered_map<K, V>>
    : type_identity<std::pair<K, V>> {};

template <typename T, typename F>
std::function<bool(T const &, T const &)> compare_by(F const &f) {
  return [=](T const &lhs, T const &rhs) { return f(lhs) < f(rhs); };
}

}

#endif
