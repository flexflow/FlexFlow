#include "pcg/strided_rectangle.h"
#include "op-attrs/dim_ordered/transform.h"
#include "pcg/device_coordinates.dtg.h"
#include "pcg/device_id_t.dtg.h"
#include "pcg/strided_rectangle_side.h"
#include "utils/containers/as_vector.h"
#include "utils/containers/product.h"
#include "utils/containers/transform.h"

namespace FlexFlow {

size_t get_num_dims(StridedRectangle const &rect) {
  return rect.sides.size();
}

num_points_t get_num_points(StridedRectangle const &rect) {
  return num_points_t{
      product(transform(rect.sides, [](StridedRectangleSide const &side) {
        return side.num_points.unwrapped;
      }))};
}

size_t get_size(StridedRectangle const &rect) {
  return product(transform(rect.sides, [](StridedRectangleSide const &side) {
    return get_side_size(side).unwrapped;
  }));
}

} // namespace FlexFlow
