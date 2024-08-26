#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_BIDICT_ALGORITHMS_BIDICT_FROM_PAIRS_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_BIDICT_ALGORITHMS_BIDICT_FROM_PAIRS_H

#include "utils/bidict/bidict.h"

namespace FlexFlow {

template <typename C,
          typename L = typename C::value_type::first_type,
          typename R = typename C::value_type::second_type>
bidict<L, R> bidict_from_pairs(C const &c) {
  return bidict<L, R>{c.begin(), c.end()};
}

} // namespace FlexFlow

#endif
