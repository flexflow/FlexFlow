#ifndef _FLEXFLOW_LIB_OP_ATTRS_INCLUDE_OP_ATTRS_DIM_ORDERED_SLICE_H
#define _FLEXFLOW_LIB_OP_ATTRS_INCLUDE_OP_ATTRS_DIM_ORDERED_SLICE_H

#include "op-attrs/dim_ordered/dim_ordered.h"
#include "utils/containers/subvec.h"
#include "utils/containers/transform.h"
#include "utils/containers/vector_of.h"
#include "utils/optional.h"

namespace FlexFlow {

template <typename Idx, typename T>
DimOrdered<Idx, T> nonoverloaded_slice(DimOrdered<Idx, T> const &d,
                                       std::optional<Idx> const &start,
                                       std::optional<Idx> const &end) {
  auto to_raw_idx = [](std::optional<Idx> const &idx) -> std::optional<int> {
    return transform(idx, [](Idx const &i) { return i.value; });
  };

  return DimOrdered<Idx, T>{
      subvec(vector_of(d), to_raw_idx(start), to_raw_idx(end))};
}

template <typename Idx, typename T>
DimOrdered<Idx, T> slice(DimOrdered<Idx, T> const &d,
                         std::optional<Idx> const &start,
                         std::optional<Idx> const &end) {
  return nonoverloaded_slice(d, start, end);
}

template <typename Idx, typename T>
DimOrdered<Idx, T> slice(DimOrdered<Idx, T> const &d,
                         std::nullopt_t const &start,
                         Idx const &end) {
  return nonoverloaded_slice(
      d, std::optional<Idx>{start}, std::optional<Idx>{end});
}

template <typename Idx, typename T>
DimOrdered<Idx, T> slice(DimOrdered<Idx, T> const &d,
                         Idx const &start,
                         std::nullopt_t const &end) {
  return nonoverloaded_slice(
      d, std::optional<Idx>{start}, std::optional<Idx>{end});
}

template <typename Idx, typename T>
DimOrdered<Idx, T>
    slice(DimOrdered<Idx, T> const &d, Idx const &start, Idx const &end) {
  return nonoverloaded_slice(
      d, std::optional<Idx>{start}, std::optional<Idx>{end});
}

} // namespace FlexFlow

#endif
