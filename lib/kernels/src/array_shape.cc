#include "kernels/array_shape.h"
#include "utils/containers.h"

namespace FlexFlow {

ArrayShape::ArrayShape(size_t *dims, size_t num_dims) {
  for (size_t i = 0; i < num_dims; i++) {
    this->dims.push_back(dims[i]);
  }
}

ArrayShape::ArrayShape(std::vector<std::size_t> const &dims)
  : dims(dims.begin(), dims.end())
{ }

std::size_t ArrayShape::get_volume() const {
  return product(this->dims);
}

}
