#include "op-attrs/ops/noop.h"

namespace FlexFlow {

TensorShape get_output_shape(NoopAttrs const &,
                             TensorShape const &input_shape) {
  return input_shape;
}

ParallelTensorShape get_output_shape(NoopAttrs const &,
                                     ParallelTensorShape const &input_shape) {
  return input_shape;
}

} // namespace FlexFlow
