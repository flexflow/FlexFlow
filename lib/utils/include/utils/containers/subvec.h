#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_SUBVEC_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_SUBVEC_H

#include <optional>
#include <vector>

namespace FlexFlow {

template <typename T>
std::vector<T> subvec(std::vector<T> const &v,
                      std::optional<int> const &maybe_start,
                      std::optional<int> const &maybe_end) {
  auto begin_iter = v.cbegin();
  auto end_iter = v.cend();

  auto resolve_loc = [&](int idx) ->
      typename std::vector<T>::iterator::difference_type {
        if (idx < 0) {
          return std::max(0, (int)v.size() + idx);
        } else {
          return std::min(idx, (int)v.size());
        }
      };

  if (maybe_start.has_value()) {
    begin_iter += resolve_loc(maybe_start.value());
  }
  if (maybe_end.has_value()) {
    end_iter = v.cbegin() + resolve_loc(maybe_end.value());
  }
  if (begin_iter >= end_iter) {
    return {};
  }

  std::vector<T> output(begin_iter, end_iter);
  return output;
}

} // namespace FlexFlow

#endif
