#ifndef _FLEXFLOW_PCG_INCLUDE_PCG_PARALLEL_TENSOR_GUID_T_H
#define _FLEXFLOW_PCG_INCLUDE_PCG_PARALLEL_TENSOR_GUID_T_H

#include "utils/graph/multidiedge.h"
#include "utils/strong_typedef.h"

namespace FlexFlow {

struct parallel_tensor_guid_t
    : strong_typedef<parallel_tensor_guid_t, MultiDiOutput> {
  using strong_typedef::strong_typedef;
};
FF_TYPEDEF_HASHABLE(parallel_tensor_guid_t);
FF_TYPEDEF_PRINTABLE(parallel_tensor_guid_t, "parallel_tensor_guid");

} // namespace FlexFlow

#endif
