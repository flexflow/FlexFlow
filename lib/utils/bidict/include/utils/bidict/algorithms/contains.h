#ifndef _FLEXFLOW_LIB_UTILS_BIDICT_INCLUDE_UTILS_BIDICT_ALGORITHMS_CONTAINS_H
#define _FLEXFLOW_LIB_UTILS_BIDICT_INCLUDE_UTILS_BIDICT_ALGORITHMS_CONTAINS_H

#include "utils/bidict/bidict.h"

namespace FlexFlow {

template <typename K, typename V>
bool contains_l(bidict<K, V> const &m, K const &k) {
  return m.find(k) != m.end();
}

template <typename K, typename V>
bool contains_r(bidict<K, V> const &m, V const &v) {
  return m.find(v) != m.end();
}

} // namespace FlexFlow

#endif
