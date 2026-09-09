#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_BIDICT_ALGORITHMS_UNORDERED_BIDICT_FROM_MAP_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_BIDICT_ALGORITHMS_UNORDERED_BIDICT_FROM_MAP_H

#include "utils/bidict/unordered_bidict.h"

namespace FlexFlow {

template <typename L, typename R>
unordered_bidict<L, R> unordered_bidict_from_map(std::map<L, R> const &m) {
  return unordered_bidict<L, R>{m.begin(), m.end()};
}

template <typename L, typename R>
unordered_bidict<L, R>
    unordered_bidict_from_map(std::unordered_map<L, R> const &m) {
  return unordered_bidict<L, R>{m.begin(), m.end()};
}

} // namespace FlexFlow

#endif
