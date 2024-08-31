#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_ENUMERATE_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_ENUMERATE_H

#include "utils/containers/enumerate_vector.h"
#include <map>
#include <unordered_set>
#include <vector>

namespace FlexFlow {

/**
 * @brief Generate a map from indices to elements of \p c.
 *
 * @note We return a <tt>std::map<int, T></tt> rather than a
 * <tt>std::vector<std::pair<int, T>></tt> for consistency
 * with enumerate(FFOrdered<T> const &). Note that <tt>std::map</tt>
 * provides ordered iteration in increasing order, so iterating through
 * the result of this function should still function as expected.
 */
template <typename T>
std::map<int, T> enumerate(std::vector<T> const &c) {
  return enumerate_vector(c);
}

/**
 * @brief Choose an arbitrary ordering of the elements of \p c and
 * return a map from indices of this ordering to elements of \p c.

 *
 * @note We return a <tt>std::map<int, T></tt> rather than a
 * <tt>std::vector<std::pair<int, T>></tt> for consistency
 * with enumerate(FFOrdered<T> const &). Note that <tt>std::map</tt>
 * provides ordered iteration in increasing order, so iterating through
 * the result of this function should still function as expected.
 */
template <typename T>
std::map<int, T> enumerate(std::unordered_set<T> const &c) {
  std::map<int, T> result;
  int idx = 0;
  for (auto const &v : c) {
    result.insert({idx++, v});
  }
  return result;
}

} // namespace FlexFlow

#endif
