#ifndef _FLEXFLOW_LIB_OP_ATTRS_INCLUDE_OP_ATTRS_DIM_ORDERED_ZIP_H
#define _FLEXFLOW_LIB_OP_ATTRS_INCLUDE_OP_ATTRS_DIM_ORDERED_ZIP_H

#include "op-attrs/dim_ordered/dim_ordered.h"
#include "utils/containers/vector_of.h"
#include "utils/containers/zip.h"

namespace FlexFlow {

template <typename Idx, typename T1, typename T2>
DimOrdered<Idx, std::pair<T1, T2>> zip(DimOrdered<Idx, T1> const &lhs,
                                       DimOrdered<Idx, T2> const &rhs) {
  return DimOrdered<Idx, std::pair<T1, T2>>{
      zip(vector_of(lhs), vector_of(rhs))};
}

} // namespace FlexFlow

#endif
