#include "op-attrs/tensor_shape.h"
#include "utils/containers.h"

namespace FlexFlow {

size_t TensorShape::get_volume() const {
  return product(dims);
}

size_t TensorShape::at(ff_dim_t d) const {
  return dims.at(d);
}

size_t TensorShape::operator[](ff_dim_t d) const {
  return dims[d];
}

} // namespace FlexFlow
