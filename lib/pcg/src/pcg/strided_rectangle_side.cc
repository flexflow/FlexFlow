#include "pcg/strided_rectangle_side.h"
#include "utils/exception.h"

namespace FlexFlow {

StridedRectangleSide strided_side_from_size_and_stride(side_size_t, int stride) {
  NOT_IMPLEMENTED();
}

side_size_t get_side_size(StridedRectangleSide const &s) {
  return s.num_points.unwrapped * s.stride;
}

}
