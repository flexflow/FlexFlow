#include "op-attrs/ops/batch_norm.h"

namespace FlexFlow {

// input: [b, c, h, w]
// output: [b, c, h, w]
ParallelTensorShape get_output_shape(BatchNormAttrs const &attrs,
                                     ParallelTensorShape const &input) {
  ParallelTensorShape output_shape = input;
  return output_shape;
}

} // namespace FlexFlow
