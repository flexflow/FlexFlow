#include "kernels/array_shape.h"
#include "utils/containers.h"

namespace FlexFlow {

ArrayShape::ArrayShape(size_t *_dims, size_t num_dims)
    : dims(_dims, _dims + num_dims) {}

std::size_t ArrayShape::get_volume() const {
  return product(this->dims);
}

std::size_t get_volume(FlexFlow::ArrayShape const&) {
  NOT_IMPLEMENTED(); 
}

} // namespace FlexFlow
