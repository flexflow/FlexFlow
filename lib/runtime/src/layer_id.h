#ifndef _FF_TYPE_H
#define _FF_TYPE_H

#include "op-attrs/ffconst.h"
#include <cstddef>
#include "utils/strong_typedef.h"

namespace FlexFlow {

struct LayerID : strong_typedef<LayerID, size_t> {
  using strong_typedef::strong_typedef; 
};

}

#endif 
