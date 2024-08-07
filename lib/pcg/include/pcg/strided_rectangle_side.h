#ifndef _FLEXFLOW_LIB_PCG_INCLUDE_PCG_STRIDED_RECTANGLE_SIDE_H
#define _FLEXFLOW_LIB_PCG_INCLUDE_PCG_STRIDED_RECTANGLE_SIDE_H

#include "pcg/side_size_t.dtg.h"
#include "pcg/stride_t.dtg.h"
#include "pcg/strided_rectangle_side.dtg.h"

namespace FlexFlow {

StridedRectangleSide strided_side_from_size_and_stride(side_size_t,
                                                       stride_t stride);

side_size_t get_side_size(StridedRectangleSide const &);

} // namespace FlexFlow

#endif
