#include "op-attrs/tensor_shape.h"

namespace FlexFlow {

size_t TensorShape::at(ff_dim_t d) const {
  return dims.at(d);
}

size_t TensorShape::operator[](ff_dim_t d) const {
  return dims[d];
}

}
