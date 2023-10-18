#include "op-attrs/ops/softmax.h"
#include "utils/exception.h"

namespace FlexFlow {

ParallelTensorShape get_output_shape(SoftmaxAttrs const &attrs,
                                     ParallelTensorShape const &input_shape) {
  if (input_shape.num_dims() < 2) {
    throw mk_runtime_error("SoftmaxAttrs: input_shape.num_dims() < 2");
  }
  return input_shape;
}

} // namespace FlexFlow
