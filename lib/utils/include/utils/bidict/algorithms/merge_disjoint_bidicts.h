#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_BIDICT_ALGORITHMS_MERGE_DISJOINT_BIDICTS_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_BIDICT_ALGORITHMS_MERGE_DISJOINT_BIDICTS_H

#include "utils/bidict/algorithms/left_entries.h"
#include "utils/bidict/algorithms/right_entries.h"
#include "utils/bidict/bidict.h"
#include "utils/containers/are_disjoint.h"
#include "utils/exception.h"

namespace FlexFlow {

template <typename K, typename V>
bidict<K, V> merge_disjoint_bidicts(bidict<K, V> const &lhs,
                                    bidict<K, V> const &rhs) {
  if (!are_disjoint(left_entries(lhs), left_entries(rhs))) {
    throw mk_runtime_error(
        fmt::format("Left entries of {} and {} are non-disjoint", lhs, rhs));
  }
  if (!are_disjoint(right_entries(lhs), right_entries(rhs))) {
    throw mk_runtime_error(
        fmt::format("Right entries of {} and {} are non-disjoint", lhs, rhs));
  }

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
