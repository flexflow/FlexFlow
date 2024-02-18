#ifndef _FLEXFLOW_LIB_UTILS_BIDICT_INCLUDE_UTILS_BIDICT_ALGORITHMS_CONTAINS_H
#define _FLEXFLOW_LIB_UTILS_BIDICT_INCLUDE_UTILS_BIDICT_ALGORITHMS_CONTAINS_H

#include "utils/bidict/bidict.h"

namespace FlexFlow {

template <typename L, typename R>
bool contains_l(bidict<L, R> const &m, L const &k) {
  return m.find_l(k) != m.end();
}

template <typename L, typename R>
bool contains_r(bidict<L, R> const &m, R const &v) {
  return m.find_r(v) != m.end();
}

} // namespace FlexFlow

#endif
