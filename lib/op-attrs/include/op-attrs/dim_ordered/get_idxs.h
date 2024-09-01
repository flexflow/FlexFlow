#ifndef _FLEXFLOW_LIB_OP_ATTRS_INCLUDE_OP_ATTRS_DIM_ORDERED_GET_IDXS_H
#define _FLEXFLOW_LIB_OP_ATTRS_INCLUDE_OP_ATTRS_DIM_ORDERED_GET_IDXS_H

#include "op-attrs/dim_ordered.h"
#include "utils/containers/transform.h"
#include "utils/containers/count.h"

namespace FlexFlow {

template <typename T>
std::vector<ff_dim_t> get_idxs(FFOrdered<T> const &d) {
  return transform(count(d.size()), [](int i) { return ff_dim_t{i}; });
}

} // namespace FlexFlow

#endif
