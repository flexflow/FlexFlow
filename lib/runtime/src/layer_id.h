#ifndef _FF_TYPE_H
#define _FF_TYPE_H

#include "utils/strong_typedef.h"

namespace FlexFlow {

struct LayerID : strong_typedef<LayerID, size_t> {
  using strong_typedef::strong_typedef;
};

} // namespace FlexFlow

MAKE_TYPEDEF_HASHABLE(::FlexFlow::LayerID);
MAKE_TYPEDEF_PRINTABLE(::FlexFlow::LayerID, "LayerID");

#endif
