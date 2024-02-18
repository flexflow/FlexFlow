#ifndef _FLEXFLOW_LIB_UTILS_BIDICT_INCLUDE_UTILS_BIDICT_ALGORITHMS_KEYS_H
#define _FLEXFLOW_LIB_UTILS_BIDICT_INCLUDE_UTILS_BIDICT_ALGORITHMS_KEYS_H

#include "utils/bidict/bidict.h"
#include <unordered_set>

namespace FlexFlow {

template <typename L, typename R>
std::unordered_set<L> keys_l(bidict<L, R> const &m) {
  std::unordered_set<L> result;
  for (auto const &[l, r] : m) {
    result.insert(l);
  }
  return result;
}

template <typename L, typename R>
std::unordered_set<R> keys_r(bidict<L, R> const &m) {
  std::unordered_set<R> result;
  for (auto const &[l, r] : m) {
    result.insert(r);
  }
  return result;
}

} // namespace FlexFlow

#endif
