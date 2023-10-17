#include "op-attrs/ops/element_unary.h"

namespace FlexFlow {

ParallelTensorShape get_output_shape(ElementUnaryAttrs const &atts,
                                     ParallelTensorShape const &input) {
  ParallelTensorShape output = input;
  return output;
}

} // namespace FlexFlow
