#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_REPLICATE_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_REPLICATE_H

#include "utils/exception.h"
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
T foldl(C const &c, T init, F func) {
  T result = init;
  for (auto const &elem : c) {
    result = func(result, elem);
  }
  return result;
}

/**
 * @brief
 * Applies `func` to the elements of `c` from left to right, accumulating the
 * result. The first element of `c` is used as the starting point for the
 * accumulation.
 *
 * @example
 *   std::vector<int> nums = {1, 2, 3, 4};
 *   int result = foldl1(nums, [](int a, int b) { return a + b; });
 *   result -> (((1+2)+3)+4) = 10
 *
 * @note
 * For more information, see
 * https://hackage.haskell.org/package/base-4.20.0.1/docs/Prelude.html#v:foldl1
 * @throws std::runtime_error if the container is empty.
 */
template <typename C, typename F, typename E = typename C::value_type>
E foldl1(C const &c, F func) {
  if (c.empty()) {
    throw mk_runtime_error(
        fmt::format("foldl1 received empty container: {}", c));
  }
  std::optional<E> result = std::nullopt;

  for (E const &e : c) {
    if (!result.has_value()) {
      result = e;
    } else {
      result = func(result.value(), e);
    }
  }
  return result.value();
}

} // namespace FlexFlow

#endif
