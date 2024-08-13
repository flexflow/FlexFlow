#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_FILTER_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_FILTER_H

#include <algorithm>
#include <map>
#include <set>
#include <unordered_set>
#include <vector>

namespace FlexFlow {

template <typename Elem, typename F>
std::vector<Elem> filter(std::vector<Elem> const &v, F const &f) {
  std::vector<Elem> result;
  std::copy_if(v.cbegin(), v.cend(), std::back_inserter(result), f);
  return result;
}

template <typename Elem, typename F>
std::unordered_set<Elem> filter(std::unordered_set<Elem> const &s, F const &f) {
  std::unordered_set<Elem> result;
  std::copy_if(s.cbegin(), s.cend(), std::inserter(result, result.begin()), f);
  return result;
}

template <typename K, typename V, typename F>
std::unordered_map<K, V> filter(std::unordered_map<K, V> const &m, F const &f) {
  std::unordered_map<K, V> result;
  std::copy_if(m.cbegin(), m.cend(), std::inserter(result, result.begin()), f);
  return result;
}

template <typename Elem, typename F>
std::set<Elem> filter(std::set<Elem> const &s, F const &f) {
  std::set<Elem> result;
  std::copy_if(s.cbegin(), s.cend(), std::inserter(result, result.begin()), f);
  return result;
}

template <typename K, typename V, typename F>
std::map<K, V> filter(std::map<K, V> const &m, F const &f) {
  std::map<K, V> result;
  std::copy_if(m.cbegin(), m.cend(), std::inserter(result, result.begin()), f);
  return result;
}

template <typename K, typename V, typename F>
std::unordered_multiset<K, V> filter(std::unordered_multiset<K, V> const &m,
                                     F const &f) {
  std::unordered_multiset<K, V> result;
  std::copy_if(m.cbegin(), m.cend(), std::inserter(result, result.begin()), f);
  return result;
}

} // namespace FlexFlow

#endif
