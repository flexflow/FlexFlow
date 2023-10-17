#include "dropout.h"
#include "op-attrs/get_output_shapes.h"

namespace FlexFlow {

ParallelTensorShape get_output_shape(DropoutAttrs const &attrs,
                                     ParallelTensorShape const &input) {
  ParallelTensorShape output = input;
  return output;
}

} // namespace FlexFlow
