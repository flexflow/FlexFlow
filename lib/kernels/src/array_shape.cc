#include "kernels/array_shape.h"
#include "utils/containers.h"

namespace FlexFlow {

ArrayShape::ArrayShape(size_t *_dims, size_t num_dims)
    : dims(_dims, _dims + num_dims) {}

std::size_t ArrayShape::get_volume() const {
  return product(this->dims);
}

} // namespace FlexFlow
