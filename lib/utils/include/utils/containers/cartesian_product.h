#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_CARTESIAN_PRODUCT_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_CARTESIAN_PRODUCT_H

#include "utils/containers/vector_of.h"
#include "utils/hash/vector.h"
#include <functional>
#include <unordered_set>
#include <vector>

namespace FlexFlow {

template <typename E>
std::unordered_set<std::vector<E>>
    cartesian_product(std::vector<std::unordered_set<E>> const &containers) {
  std::unordered_set<std::vector<E>> result;

  std::function<void(std::vector<E> &, size_t)> recurse =
      [&](std::vector<E> &current, size_t depth) {
        if (depth == containers.size()) {
          result.insert(current);
          return;
        }

        for (E const &item : containers.at(depth)) {
          current.push_back(item);
          recurse(current, depth + 1);
          current.pop_back();
        }
      };

  std::vector<E> current;
  recurse(current, 0);

  return result;
}

} // namespace FlexFlow

#endif
