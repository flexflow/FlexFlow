#include "op-attrs/ops/element_unary.h"

namespace FlexFlow {

ParallelTensorShape get_output_shape(ElementUnaryAttrs const &atts,
                                     ParallelTensorShape const &input_shape) {

  return input_shape;
}

} // namespace FlexFlow
