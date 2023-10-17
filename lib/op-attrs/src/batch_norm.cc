#include "op-attrs/ops/batch_norm.h"

namespace FlexFlow {

bool is_valid(BatchNormAttrs const &attrs, ParallelTensorShape const &input) {
  if (input.num_dims() != 4) {
    return false;
  }
  return true;
}

// input: [b, c, h, w]
// output: [b, c, h, w]
ParallelTensorShape get_output_shape(BatchNormAttrs const &attrs,
                                     ParallelTensorShape const &input) {
  ParallelTensorShape output_shape = input;
  return output_shape;
}

} // namespace FlexFlow
