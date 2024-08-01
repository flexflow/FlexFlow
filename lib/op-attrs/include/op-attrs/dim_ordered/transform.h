#ifndef _FLEXFLOW_LIB_OP_ATTRS_INCLUDE_OP_ATTRS_DIM_ORDERED_TRANSFORM_H
#define _FLEXFLOW_LIB_OP_ATTRS_INCLUDE_OP_ATTRS_DIM_ORDERED_TRANSFORM_H

#include "op-attrs/dim_ordered.h"
#include "utils/containers/as_vector.h"
#include "utils/containers/vector_transform.h"

namespace FlexFlow {

template <typename Idx, typename T, typename F>
DimOrdered<Idx, std::invoke_result_t<F, T>>
    transform(DimOrdered<Idx, T> const &d, F f) {
  using Out = std::invoke_result_t<F, T>;

  return DimOrdered<Idx, Out>{vector_transform(as_vector(d), f)};
}

} // namespace FlexFlow

#endif
