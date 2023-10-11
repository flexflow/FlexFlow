#include "op-attrs/ops/element_unary.h"

namespace FlexFlow {

bool ElementUnaryAttrs::is_valid(ParallelTensorShape const &input) const {
  if (!input.is_valid()) {
    return false;
  }
  return true;
}

ParallelTensorShape get_output_shape(ElementUnaryAttrs const &atts,
                                     ParallelTensorShape const &input) {
  ParallelTensorShape output = input;
  return output;
}

} // namespace FlexFlow
