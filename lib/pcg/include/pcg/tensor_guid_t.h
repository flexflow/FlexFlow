#ifndef _FLEXFLOW_PCG_INCLUDE_PCG_TENSOR_GUID_T_H
#define _FLEXFLOW_PCG_INCLUDE_PCG_TENSOR_GUID_T_H

#include "utils/graph.h"
#include "utils/strong_typedef.h"

namespace FlexFlow {

struct weight_guid_t : public strong_typedef<weight_guid_t, MultiDiOutput> {
  using strong_typedef::strong_typedef;
};
FF_TYPEDEF_HASHABLE(weight_guid_t);
FF_TYPEDEF_PRINTABLE(weight_guid_t, "weight_guid");

struct tensor_guid_t : public strong_typedef<tensor_guid_t, MultiDiOutput> {
  using strong_typedef::strong_typedef;
};
FF_TYPEDEF_HASHABLE(tensor_guid_t);
FF_TYPEDEF_PRINTABLE(tensor_guid_t, "tensor_guid");

} // namespace FlexFlow

#endif
