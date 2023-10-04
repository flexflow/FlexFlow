#include "pcg/parallel_tensor.h"

namespace FlexFlow {

size_t ParallelTensor::get_volume() const {
  NOT_IMPLEMENTED();
}

ParallelTensorShape ParallelTensor::get_shape() const {
  return {dims, data_type};
}

int ParallelTensor::num_dims() const {
  NOT_IMPLEMENTED();
}

ParallelTensor::operator ParallelTensorShape() const {
  return get_shape();
}

} // namespace FlexFlow
