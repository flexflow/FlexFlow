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

} // namespace FlexFlow
