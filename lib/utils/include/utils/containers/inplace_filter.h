#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_INPLACE_FILTER_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_INPLACE_FILTER_H

#include "utils/containers/filter.h"
#include <algorithm>
#include <unordered_set>
#include <vector>

namespace FlexFlow {

template <typename Elem, typename F>
void inplace_filter(std::vector<Elem> &v, F const &f) {
  v.erase(
      std::remove_if(v.begin(), v.end(), [&](Elem const &e) { return !f(e); }),
      v.end());
}

template <typename Elem, typename F>
void inplace_filter(std::unordered_set<Elem> &s, F const &f) {
  s = filter(s, f);
}

template <typename Elem, typename F>
void inplace_filter(std::set<Elem> &s, F const &f) {
  s = filter(s, f);
}

template <typename K, typename V, typename F>
void inplace_filter(std::unordered_map<K, V> &s, F const &f) {
  s = filter(s, f);
}

template <typename K, typename V, typename F>
void inplace_filter(std::map<K, V> &s, F const &f) {
  s = filter(s, f);
}

} // namespace FlexFlow

#endif
