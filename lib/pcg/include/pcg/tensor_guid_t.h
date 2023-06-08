#ifndef _FLEXFLOW_PCG_INCLUDE_PCG_TENSOR_GUID_T_H
#define _FLEXFLOW_PCG_INCLUDE_PCG_TENSOR_GUID_T_H

#include "utils/graph.h"

namespace FlexFlow {

struct tensor_guid_t : strong_typedef<tensor_guid_t, MultiDiOutput> {
  using strong_typedef::strong_typedef;
};

} // namespace FlexFlow

MAKE_TYPEDEF_PRINTABLE(::FlexFlow::tensor_guid_t, "tensor_guid");
MAKE_TYPEDEF_HASHABLE(::FlexFlow::tensor_guid_t);

#endif
