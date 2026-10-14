#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_REPLICATE_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_REPLICATE_H

#include "utils/fmt/vector.h"
#include <cassert>
#include <fmt/format.h>
#include <iterator>
#include <optional>

namespace FlexFlow {

/**
 * @brief
 * Iteratively applies `func` to the elements of `c` from left to right.
 * `init` is used as the starting value.
 *
 * @example
 *   std::vector<int> nums = {1, 2, 3, 4};
 *   int result = foldl(nums, 0, [](int a, int b) { return a + b; });
 *   result -> ((((0+1)+2)+3)+4) = 10
 *
 * @note
 * For more information, see
 * https://hackage.haskell.org/package/base-4.20.0.1/docs/Prelude.html#v:foldl
 */
template <typename C, typename T, typename F>
T foldl(C const &c, T const &init, F func) {
  T result = init;
  for (auto const &elem : c) {
    result = func(result, elem);
  }
  return result;
}

} // namespace FlexFlow

#endif
