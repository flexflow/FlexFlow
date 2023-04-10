#ifndef _FF_TYPE_H
#define _FF_TYPE_H

#include "op-attrs/ffconst.h"
#include <cstddef>
#include "utils/strong_typedef.h"
#include "utils/fmt.h"

namespace FlexFlow {

struct LayerID : strong_typedef<LayerID, size_t> {
  using strong_typedef::strong_typedef; 
};

}

MAKE_TYPEDEF_HASHABLE(::FlexFlow::LayerID);
MAKE_TYPEDEF_PRINTABLE(::FlexFlow::LayerID, "LayerID");

#endif 
