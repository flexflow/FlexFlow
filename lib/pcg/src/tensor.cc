#include "pcg/tensor.h"

namespace FlexFlow {

size_t Tensor::get_num_elements() const {
  return this->get_shape().get_num_elements();
}

Tensor::operator TensorShape() const {
  return TensorShape{dims, data_type};
}

TensorShape Tensor::get_shape() const {
  return TensorShape(*this);
}

Tensor construct_tensor_from_output_shape(TensorShape const &shape) {
  return Tensor{shape.dims, shape.data_type, std::nullopt, false, std::nullopt};
}

} // namespace FlexFlow
