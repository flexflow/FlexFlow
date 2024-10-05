#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_GET_ALL_PERMUTATIONS_WITH_REPETITION_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_GET_ALL_PERMUTATIONS_WITH_REPETITION_H

#include <unordered_set>
#include <vector>

namespace FlexFlow {

/**
 * @brief For a given container `c` and integer `n`, return all possible vectors
 * of size `n` that only contain (possibly duplicated) elements of `c`.
 * @details
 * https://en.wikipedia.org/wiki/Permutation#Permutations_with_repetition
 **/
template <typename C, typename T = typename C::value_type>
std::unordered_multiset<std::vector<T>>
    get_all_permutations_with_repetition(C const &container, int n) {
  std::unordered_multiset<std::vector<T>> result;

  if (container.empty() || n == 0) {
    return result;
  }

  std::vector<T> elements(std::begin(container), std::end(container));
  std::vector<int> indices(n, 0);

  while (true) {
    std::vector<T> perm(n);
    for (int i = 0; i < n; ++i) {
      perm[i] = elements[indices[i]];
    }
    result.insert(perm);

    int i = n - 1;
    while (i != -1 && ++indices[i] == elements.size()) {
      indices[i] = 0;
      --i;
    }

    if (i == -1) {
      break;
    }
  }

  return result;
}

} // namespace FlexFlow

#endif
