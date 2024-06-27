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

StridedRectangleSide StridedRectangle::at(ff_dim_t const &dim) const {
  StridedRectangleSide side = this->sides.at(dim);
  return side;
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

num_points_t StridedRectangleSide::get_num_points() const {
  return num_points;
}

size_t StridedRectangle::num_dims() const {
  return this->sides.size();
}

} // namespace FlexFlow
