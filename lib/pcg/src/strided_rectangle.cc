#include "pcg/strided_rectangle.h"
#include "op-attrs/dim_ordered/transform.h"
#include "utils/containers.h"

namespace FlexFlow {

/* size_t StridedRectangle::at(FFOrdered<num_points_t> const &coord) const { */
/*   assert(coord.size() == this->num_dims()); */

/*   size_t _1d_stride = 1; */
/*   size_t idx = 0; */
/*   for (auto dim : inner_to_outer_idxs(this->sides)) { */
/*     idx += this->sides.at(dim).at(coord.at(dim)).value() * _1d_stride; */
/*     _1d_stride *= this->sides.at(dim).get_size().value(); */
/*   } */
/*   return idx; */
/* } */

size_t get_num_dims(StridedRectangle const &rect) {
  return rect.sides.size();
}

num_points_t get_num_points(StridedRectangle const &rect) {
  return num_points_t{
      product(transform(rect.sides, [](StridedRectangleSide const &side) {
        return side.num_points.unwrapped;
      }))};
}

StridedRectangleSide get_side_at_idx(StridedRectangle const &rect,
                                     ff_dim_t const &idx) {
  return rect.sides.at(idx);
}

} // namespace FlexFlow

