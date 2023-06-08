#ifndef _FLEXFLOW_PCG_INCLUDE_PCG_OPERATOR_GUID_T_H
#define _FLEXFLOW_PCG_INCLUDE_PCG_OPERATOR_GUID_T_H

#include "utils/graph.h"
#include "utils/strong_typedef.h"

namespace FlexFlow {

struct operator_guid_t : strong_typedef<operator_guid_t, Node> {
  using strong_typedef::strong_typedef;
};

} // namespace FlexFlow

MAKE_TYPEDEF_PRINTABLE(::FlexFlow::operator_guid_t, "operator_guid");
MAKE_TYPEDEF_HASHABLE(::FlexFlow::operator_guid_t);

#endif
