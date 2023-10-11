#include "op-attrs/ops/softmax.h"

namespace FlexFlow {

bool SoftmaxAttrs::is_valid(ParallelTensorShape const &input) const {
  if (!input.is_valid()) {
    return false;
  }
  if (input.num_dims() < 2) {
    return false;
  }
  return true;
}

ParallelTensorShape get_output_shape(SoftmaxAttrs const &attrs,
                                     ParallelTensorShape const &input) {
  assert(attrs.is_valid(input));
  ParallelTensorShape output = input;
  return output;
}

} // namespace FlexFlow
