#include "op-attrs/ops/element_binary.h"

namespace FlexFlow {

ParallelTensorShape get_output_shape(ElementBinaryAttrs const &,
                                     ParallelTensorShape const &,
                                     ParallelTensorShape const &) {
  NOT_IMPLEMENTED();
}

TensorShape get_output_shape(ElementBinaryAttrs const &,
                             TensorShape const &,
                             TensorShape const &) {
  NOT_IMPLEMENTED();
}

} // namespace FlexFlow
