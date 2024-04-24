#ifndef _FLEXFLOW_PCG_INCLUDE_PCG_STRIDED_RECTANGLE_H
#define _FLEXFLOW_PCG_INCLUDE_PCG_STRIDED_RECTANGLE_H

#include "op-attrs/ff_dim.dtg.h"
#include "pcg/side_size_t.dtg.h"
#include "pcg/strided_rectangle.dtg.h"

namespace FlexFlow {

size_t get_num_dims(StridedRectangle const &);
StridedRectangleSide get_side_at_idx(StridedRectangle const &, ff_dim_t const &);

} // namespace FlexFlow

#endif
