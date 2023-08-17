#ifndef _FLEXFLOW_PCG_INCLUDE_PCG_LAYER_GUID_T_H
#define _FLEXFLOW_PCG_INCLUDE_PCG_LAYER_GUID_T_H

#include "utils/strong_typedef.h"
#include "utils/graph.h"

namespace FlexFlow {

struct layer_guid_t : public strong_typedef<layer_guid_t, Node> {
  using strong_typedef::strong_typedef;
};
FF_TYPEDEF_HASHABLE(layer_guid_t);
FF_TYPEDEF_PRINTABLE(layer_guid_t, "layer_guid");

}

#endif
