#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_SCANL_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_SCANL_H

#include <optional>
#include <vector>

namespace FlexFlow {

/**
 * @brief
 * Applies `op` to the elements of `c` from left to right, accumulating
 * the intermediate results in a vector. `init` is used as the starting point
 * for the accumulation.
 *
 * @example
 *   std::vector<int> nums = {1, 2, 3, 4};
 *   auto result = scanl(nums, 0, [](int a, int b) {return a+b;});
 *   result -> {0,1,3,6,10}
 *
 * @note
 * Essentially a foldl which stores the intermediate results
 * For more information, see
 * https://hackage.haskell.org/package/base-4.20.0.1/docs/Prelude.html#v:scanl
 */
template <typename C, typename F, typename T>
std::vector<T> scanl(C const &c, T init, F const &op) {
  std::vector<T> result;

  result.push_back(init);
  for (auto const &elem : c) {
    init = op(init, elem);
    result.push_back(init);
  }

  return result;
}

/**
 * @brief
 * Applies `op` to the elements of `c` from left to right, accumulating
 * the intermediate results in a vector. The first item of `c` is used as the
 * starting point for the accumulation.
 *
 * @example
 *   std::vector<int> nums = {1, 2, 3, 4};
 *   auto result = scanl1(nums, [](int a, int b) {return a+b;});
 *   result -> {1,3,6,10}
 *
 * @note
 * Essentially a foldl1 which stores the intermediate results.
 * For more information, see
 * https://hackage.haskell.org/package/base-4.20.0.1/docs/Prelude.html#v:scanl1
 */
template <typename C, typename F, typename T = typename C::value_type>
std::vector<T> scanl1(C const &c, F op) {

  if (c.empty()) {
    return std::vector<T>();
  }

  std::optional<T> init = std::nullopt;
  std::vector<T> result;

  for (T const &elem : c) {
    if (!init.has_value()) {
      init = elem;
    } else {
      init = op(init.value(), elem);
    }
    result.push_back(init.value());
  }
  return result;
}

} // namespace FlexFlow

#endif
