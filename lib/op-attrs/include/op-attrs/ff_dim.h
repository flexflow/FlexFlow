#ifndef _FLEXFLOW_OPATTRS_INCLUDE_FF_DIM_H
#define _FLEXFLOW_OPATTRS_INCLUDE_FF_DIM_H

#include "utils/strong_typedef.h"
#include <ostream>

namespace FlexFlow {

struct ff_dim_t : public numerical_typedef<ff_dim_t, int> {
  using numerical_typedef::numerical_typedef;
};

} // namespace FlexFlow

MAKE_TYPEDEF_HASHABLE(::FlexFlow::ff_dim_t);
MAKE_TYPEDEF_PRINTABLE(::FlexFlow::ff_dim_t, "ff_dim");

#endif
