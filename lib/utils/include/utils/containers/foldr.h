#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_FOLDR_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_FOLDR_H

#include <vector>

namespace FlexFlow {

/**
 * @brief
 * Iteratively applies `func` to the elements of `c` from right to left.
 * `init` is used as the starting value.
 *
 * @example
 *   std::vector<int> nums = {1, 2, 3, 4};
 *   int result = foldl(nums, 0, [](int a, int b) { return a + b; });
 *   result -> (0+(1+(2+(3+4)))) = 10
 *
 * @note
 * For more information, see
 * https://hackage.haskell.org/package/base-4.20.0.1/docs/Prelude.html#v:foldr
 */
template <typename C, typename T, typename F>
T foldr(C const &c, T const &init, F func) {
  T result = init;
  for (auto const &elem : c) {
    result = func(result, elem);
  }
  return result;
}

} // namespace FlexFlow

#endif
