#include "op-attrs/ops/dropout.h"

namespace FlexFlow {

TensorShape get_output_shape(DropoutAttrs const &,
                             TensorShape const &) {
  NOT_IMPLEMENTED();
}

ParallelTensorShape get_output_shape(DropoutAttrs const &,
                                     ParallelTensorShape const &) {
  NOT_IMPLEMENTED();
}

} // namespace FlexFlow
