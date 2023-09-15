#include "op-attrs/ops/dropout.h"

namespace FlexFlow {

ParallelTensorShape get_output_shape(DropoutAttrs const &attrs,
                                     ParallelTensorShape const &input) {
  ParallelTensorShape output = input;
  return output;
}

} // namespace FlexFlow
