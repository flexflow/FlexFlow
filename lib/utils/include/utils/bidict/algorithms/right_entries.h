#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_BIDICT_ALGORITHMS_RIGHT_ENTRIES_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_BIDICT_ALGORITHMS_RIGHT_ENTRIES_H

#include "utils/bidict/bidict.h"
#include <unordered_set>

namespace FlexFlow {

template <typename L, typename R>
std::unordered_set<R> right_entries(bidict<L, R> const &b) {
  std::unordered_set<R> result;
  for (auto const &[l, r] : b) {
    result.insert(r);
  }
  return result;
}

} // namespace FlexFlow

#endif
