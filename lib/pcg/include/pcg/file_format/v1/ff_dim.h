#ifndef _FLEXFLOW_PCG_INCLUDE_PCG_FILE_FORMAT_V1_FF_DIM_H
#define _FLEXFLOW_PCG_INCLUDE_PCG_FILE_FORMAT_V1_FF_DIM_H

#include "op-attrs/ff_dim.h"

namespace FlexFlow {

// ff_dim_t is a strong typedef of int. This is unlikely to change, but if it
// does, this signature will need to be updated.
int to_v1(ff_dim_t const &t);
ff_dim_t from_v1(int const &vt);

} // namespace FlexFlow

#endif
