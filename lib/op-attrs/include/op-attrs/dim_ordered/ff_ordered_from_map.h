#ifndef _FLEXFLOW_LIB_OP_ATTRS_INCLUDE_OP_ATTRS_DIM_ORDERED_FF_ORDERED_FROM_MAP_H
#define _FLEXFLOW_LIB_OP_ATTRS_INCLUDE_OP_ATTRS_DIM_ORDERED_FF_ORDERED_FROM_MAP_H

#include "op-attrs/dim_ordered/dim_ordered.h"
#include "op-attrs/dim_ordered/ff_ordered_of.h"

namespace FlexFlow {

template <typename T>
FFOrdered<T> ff_ordered_from_map(std::map<ff_dim_t, T> const &m) {
  std::vector<T> raw;
  for (int i = 0; i < m.size(); i++) {
    raw.push_back(m.at(ff_dim_t{i}));
  }
  return ff_ordered_of(raw);
}

template <typename T>
FFOrdered<T> ff_ordered_from_map(std::unordered_map<ff_dim_t, T> const &m) {
  std::vector<T> raw;
  for (int i = 0; i < m.size(); i++) {
    raw.push_back(m.at(ff_dim_t{i}));
  }
  return ff_ordered_of(raw);
}

} // namespace FlexFlow

#endif
