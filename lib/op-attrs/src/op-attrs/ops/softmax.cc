#include "op-attrs/ops/softmax.h"

namespace FlexFlow {

TensorShape get_output_shape(SoftmaxAttrs const &attrs,
                             TensorShape const &input_shape) {
  NOT_IMPLEMENTED();
}

ParallelTensorShape get_output_shape(SoftmaxAttrs const &attrs,
                                     ParallelTensorShape const &input_shape) {
  NOT_IMPLEMENTED();
}

} // namespace FlexFlow
