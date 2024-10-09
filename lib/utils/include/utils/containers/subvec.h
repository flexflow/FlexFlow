#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_SUBVEC_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_SUBVEC_H

#include "utils/exception.h"
#include <optional>
#include <stdexcept>
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
        int size = static_cast<int>(v.size());
        int new_idx = idx;
        if (idx < 0) {
          new_idx = size + idx;
        }
        if (new_idx < 0 || new_idx > size) {
          throw mk_runtime_error("Index {} is out of bounds for array {}");
        }
        return new_idx;
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
