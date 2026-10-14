#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_FIND_INDEX_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_FIND_INDEX_H

#include "utils/nonnegative_int/nonnegative_range.h"
#include "utils/nonnegative_int/num_elements.h"
#include <optional>
#include <vector>

namespace FlexFlow {

template <typename T, typename F>
std::optional<nonnegative_int> find_index(std::vector<T> const &v, F &&f) {
  for (nonnegative_int idx : nonnegative_range(num_elements(v))) {
    bool matches_condition = f(v.at(idx.int_from_nonnegative_int()));
    if (matches_condition) {
      return idx;
    }
  }

  return std::nullopt;
}

} // namespace FlexFlow

#endif
