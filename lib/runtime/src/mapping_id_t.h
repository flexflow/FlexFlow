#ifndef _FLEXFLOW_RUNTIME_SRC_MAPPING_ID_T_H
#define _FLEXFLOW_RUNTIME_SRC_MAPPING_ID_T_H

#include "utils/strong_typedef.h"

namespace FlexFlow {

struct mapping_id_t : strong_typedef<mapping_id_t, size_t> {
  using strong_typedef::strong_typedef;
};

} // namespace FlexFlow

MAKE_TYPEDEF_HASHABLE(::FlexFlow::mapping_id_t);
MAKE_TYPEDEF_PRINTABLE(::FlexFlow::mapping_id_t, "mapping_id");

#endif
