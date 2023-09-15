#include "op-attrs/ops/element_unary.h"

namespace FlexFlow {

ParallelTensorShape get_output_shape(ElementUnaryAttrs const &attrs,
                                     ParallelTensorShape const &in) {
  ParallelTensorShape out = in;
  return out;
}

} // namespace FlexFlow
