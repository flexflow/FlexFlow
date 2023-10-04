#include "pcg/tensor.h"

namespace FlexFlow {
size_t Tensor::get_volume() const {
  NOT_IMPLEMENTED();
}

TensorShape Tensor::get_shape() const {
  return {dims, data_type};
}

int Tensor::num_dims() const {
  NOT_IMPLEMENTED();
}

Tensor::operator TensorShape() const {
  return get_shape();
}

} // namespace FlexFlow
