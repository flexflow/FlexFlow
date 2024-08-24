#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_BIDICT_ALGORITHMS_MERGE_BIDICTS_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_BIDICT_ALGORITHMS_MERGE_BIDICTS_H

#include "utils/bidict/bidict.h"
#include "utils/containers/are_disjoint.h"
#include "utils/bidict/algorithms/left_entries.h"
#include "utils/bidict/algorithms/right_entries.h"

namespace FlexFlow {

template <typename K, typename V>
bidict<K, V> merge_bidicts(bidict<K, V> const &lhs, bidict<K, V> const &rhs) {
  assert(are_disjoint(left_entries(lhs), left_entries(rhs)));
  assert(are_disjoint(right_entries(lhs), right_entries(rhs)));

  bidict<K, V> result;
  for (auto const &kv : lhs) {
    result.equate(kv.first, kv.second);
  }
  for (auto const &kv : rhs) {
    result.equate(kv.first, kv.second);
  }

  return result;
}

} // namespace FlexFlow

#endif
