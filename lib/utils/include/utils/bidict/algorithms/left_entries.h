#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_BIDICT_ALGORITHMS_LEFT_ENTRIES_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_BIDICT_ALGORITHMS_LEFT_ENTRIES_H

#include "utils/bidict/bidict.h"
#include <unordered_set>

namespace FlexFlow {

template <typename L, typename R>
std::unordered_set<L> left_entries(bidict<L, R> const &b) {
  std::unordered_set<L> result;
  for (auto const &[l, _] : b) {
    result.insert(l);
  }
  return result;
}

} // namespace FlexFlow

#endif
