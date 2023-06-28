#ifndef _FLEXFLOW_PCG_INCLUDE_PCG_PARALLEL_TENSOR_GUID_T_H
#define _FLEXFLOW_PCG_INCLUDE_PCG_PARALLEL_TENSOR_GUID_T_H

#include "utils/graph/labelled_graph_interfaces.h"
#include "utils/strong_typedef.h"

namespace FlexFlow {

struct parallel_tensor_guid_t
    : strong_typedef<parallel_tensor_guid_t, MultiDiOutput> {
  using strong_typedef::strong_typedef;
};

} // namespace FlexFlow

MAKE_TYPEDEF_PRINTABLE(::FlexFlow::parallel_tensor_guid_t,
                       "parallel_tensor_guid");
MAKE_TYPEDEF_HASHABLE(::FlexFlow::parallel_tensor_guid_t);

#endif
