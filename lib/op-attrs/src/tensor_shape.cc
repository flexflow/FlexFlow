#include "op-attrs/tensor_shape.h"

namespace FlexFlow {

size_t TensorShape::at(ff_dim_t) const {
  NOT_IMPLEMENTED();
}

size_t TensorShape::operator[](ff_dim_t) const {
  NOT_IMPLEMENTED();
}

} // namespace FlexFlow
