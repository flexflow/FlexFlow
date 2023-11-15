#include "pcg/strided_rectangle.h"
#include "utils/containers.h"

namespace FlexFlow {

size_t StridedRectangle::at(FFOrdered<num_points_t> const &coord) const {
  assert(coord.size() == this->num_dims());

  size_t _1d_stride = 1;
  size_t idx = 0;
  for (auto dim : inner_to_outer_idxs(this->sides)) {
    idx += this->sides.at(dim).at(coord.at(dim)).value() * _1d_stride;
    _1d_stride *= this->sides.at(dim).get_size().value();
  }
  return idx;
}

StridedRectangleSide::StridedRectangleSide(side_size_t const &num, int stride)
    : num_points(num.value()), stride(stride) {}

side_size_t StridedRectangleSide::at(num_points_t) const {
  NOT_IMPLEMENTED();
}

num_points_t StridedRectangleSide::at(side_size_t) const {
  NOT_IMPLEMENTED();
}

side_size_t StridedRectangleSide::get_size() const {
  NOT_IMPLEMENTED();
}

// StridedRectangle::StridedRectangle(
//     std::vector<StridedRectangleSide> const &sides)
//     : sides(sides) {}

} // namespace FlexFlow
