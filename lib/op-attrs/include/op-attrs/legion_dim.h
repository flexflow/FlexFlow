#ifndef _FLEXFLOW_OPATTRS_INCLUDE_LEGION_DIM_H
#define _FLEXFLOW_OPATTRS_INCLUDE_LEGION_DIM_H

#include "utils/strong_typedef.h"
#include <ostream>

namespace FlexFlow {

struct legion_dim_t : strong_typedef<legion_dim_t, int> {
  using strong_typedef::strong_typedef;

  friend std::ostream &operator<<(std::ostream &s, legion_dim_t dim) {
    return s << "legion_dim_t(" << dim.value() << ")";
  }
};

}

MAKE_TYPEDEF_HASHABLE(::FlexFlow::legion_dim_t);

#endif
