#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_CARTESIAN_PRODUCT_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_CARTESIAN_PRODUCT_H

#include "utils/containers/as_vector.h"
#include "utils/hash/vector.h"
#include <functional>
#include <unordered_set>
#include <vector>

namespace FlexFlow {

template <typename Container>
auto cartesian_product(Container const &containers) {
  using ValueType = typename Container::value_type::value_type;
  using VectorType = std::vector<ValueType>;
  using SetType = std::unordered_multiset<VectorType>;
  auto ordered = as_vector(containers);
  SetType result;

  std::function<void(VectorType &, size_t)> recurse = [&](VectorType &current,
                                                          std::size_t depth) {
    if (depth == ordered.size()) {
      result.insert(current);
      return;
    }

    for (const auto &item : ordered[depth]) {
      current.push_back(item);
      recurse(current, depth + 1);
      current.pop_back();
    }
  };

  VectorType current;
  recurse(current, 0);

  return result;
}

} // namespace FlexFlow

#endif
