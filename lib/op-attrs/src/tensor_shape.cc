#include "op-attrs/tensor_shape.h"

namespace FlexFlow {

size_t TensorShape::at(ff_dim_t dim) const {
  return dims.at(dim);
}

size_t TensorShape::operator[](ff_dim_t dim) const {
  return at(dim);
}

} // namespace FlexFlow
