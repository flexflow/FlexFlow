#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_MERGE_MAPS_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_MERGE_MAPS_H

#include "utils/containers/are_disjoint.h"
#include "utils/containers/keys.h"
#include "utils/exception.h"
#include <cassert>
#include <unordered_map>

namespace FlexFlow {

template <typename K, typename V>
std::unordered_map<K, V> merge_maps(std::unordered_map<K, V> const &lhs,
                                    std::unordered_map<K, V> const &rhs) {
  if (!are_disjoint(keys(lhs), keys(rhs))) {
    throw mk_runtime_error(
        "Key sets of merge_maps parameters are non-disjoint");
  }

  std::unordered_map<K, V> result;
  for (auto const &kv : lhs) {
    result.insert(kv);
  }
  for (auto const &kv : rhs) {
    result.insert(kv);
  }

  return result;
}

} // namespace FlexFlow

#endif
