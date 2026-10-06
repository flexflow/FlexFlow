#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_BIDICT_ALGORITHMS_RIGHT_ENTRIES_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_BIDICT_ALGORITHMS_RIGHT_ENTRIES_H

#include "utils/bidict/bidict.h"
#include <set>

namespace FlexFlow {

template <typename L, typename R>
std::set<R> right_entries(bidict<L, R> const &b) {
  return b.right_values();
}

} // namespace FlexFlow

#endif
