#ifndef _FLEXFLOW_LIB_UTILS_ALGORITHMS_INCLUDE_UTILS_ALGORITHMS_VECTOR_SLICE_H
#define _FLEXFLOW_LIB_UTILS_ALGORITHMS_INCLUDE_UTILS_ALGORITHMS_VECTOR_SLICE_H

namespace FlexFlow {

template <typename T>
std::vector<T> slice(std::vector<T> const &v,
                     optional<int> const &maybe_start,
                     optional<int> const &maybe_end) {
  auto begin_iter = v.cbegin();
  auto end_iter = v.cend();

  auto resolve_loc = [&](int idx) ->
      typename std::vector<T>::iterator::difference_type {
        if (idx < 0) {
          return v.size() - idx;
        } else {
          return idx;
        }
      };

  if (maybe_start.has_value()) {
    begin_iter += resolve_loc(maybe_start.value());
  }
  if (maybe_end.has_value()) {
    end_iter = v.cbegin() + resolve_loc(maybe_end.value());
  }

  std::vector<T> output(begin_iter, end_iter);
  return output;
}


} // namespace FlexFlow

#endif
