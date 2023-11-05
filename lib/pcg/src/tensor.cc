#include "pcg/tensor.h"

namespace FlexFlow {

Tensor::operator TensorShape() const {
  return TensorShape{dims, data_type};
}

TensorShape Tensor::get_shape() const {
  return TensorShape(*this);
}

} // namespace FlexFlow
