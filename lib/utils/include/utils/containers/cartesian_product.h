#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_CARTESIAN_PRODUCT_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_CARTESIAN_PRODUCT_H

#include "utils/containers/as_vector.h"
#include "utils/hash/vector.h"
#include <functional>
#include <unordered_set>
#include <vector>

namespace FlexFlow {

template <typename C, typename V = std::vector<typename C::value_type>>
auto cartesian_product(std::vector<C> const &containers) {
  std::unordered_multiset<V> result;

  std::function<void(V &, size_t)> recurse = [&](V &current, size_t depth) {
    if (depth == containers.size()) {
      result.insert(current);
      return;
    }

    for (const auto &item : containers.at(depth)) {
      current.push_back(item);
      recurse(current, depth + 1);
      current.pop_back();
    }
  };

  V current;
  recurse(current, 0);

  return result;
}

} // namespace FlexFlow

#endif
