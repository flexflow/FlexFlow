#include "op-attrs/ops/batch_norm.h"

namespace FlexFlow {

TensorShape get_output_shape(BatchNormAttrs const &,
                             TensorShape const &input_shape) {
  return input_shape;
}

ParallelTensorShape get_output_shape(BatchNormAttrs const &,
                                     ParallelTensorShape const &) {
  NOT_IMPLEMENTED();
}

} // namespace FlexFlow
