#include "pcg/strided_rectangle_side.h"
#include "utils/exception.h"

namespace FlexFlow {

StridedRectangleSide strided_side_from_size_and_stride(side_size_t side_size,
                                                       int stride) {
  assert((side_size.unwrapped % stride) == 0);
  return StridedRectangleSide{num_points_t{side_size.unwrapped / stride},
                              stride};
}

side_size_t get_side_size(StridedRectangleSide const &s) {
  return side_size_t{s.num_points.unwrapped * s.stride};
}

} // namespace FlexFlow
