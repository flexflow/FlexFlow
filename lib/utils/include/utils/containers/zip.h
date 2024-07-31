#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_ZIP_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_ZIP_H

#include <utility>
#include <vector>

namespace FlexFlow {

template <typename L, typename R>
std::vector<std::pair<L, R>> zip(std::vector<L> const &l,
                                 std::vector<R> const &r) {
  std::vector<std::pair<L, R>> result;
  for (int i = 0; i < std::min(l.size(), r.size()); i++) {
    result.push_back(std::make_pair(l.at(i), r.at(i)));
  }
  return result;
}

} // namespace FlexFlow

#endif
