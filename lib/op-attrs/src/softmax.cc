#include "op-attrs/ops/softmax.h"
#include "utils/exception.h"

namespace FlexFlow {

ParallelTensorShape get_output_shape(SoftmaxAttrs const &attrs,
                                     ParallelTensorShape const &input) {
  if (input.num_dims() < 2) {
    throw mk_runtime_error("SoftmaxAttrs: input.num_dims() < 2");
  }
  ParallelTensorShape output = input;
  return output;
}

} // namespace FlexFlow
