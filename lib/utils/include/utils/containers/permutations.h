#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_PERMUTATIONS_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_PERMUTATIONS_H

#include "utils/containers/sorted.h"
#include "utils/hash/vector.h"
#include <algorithm>
#include <unordered_set>
#include <vector>

namespace FlexFlow {

template <typename C, typename V = std::vector<typename C::value_type>>
auto permutations(C const &container) {
  std::unordered_set<V> result;

  V elements = sorted(container);

  result.insert(elements);

  while (std::next_permutation(elements.begin(), elements.end())) {
    result.insert(elements);
  }

  return result;
}

} // namespace FlexFlow

#endif
