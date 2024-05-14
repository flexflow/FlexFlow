#include "op-attrs/ops/element_unary.h"

namespace FlexFlow {

ParallelTensorShape get_output_shape(ElementUnaryAttrs const &, ParallelTensorShape const &) {
  NOT_IMPLEMENTED();
}

TensorShape get_output_shape(ElementUnaryAttrs const &, TensorShape const &) {
  NOT_IMPLEMENTED();
}

ParallelTensorShape get_output_shape(ElementScalarUnaryAttrs const &, ParallelTensorShape const &) {
  NOT_IMPLEMENTED();
}

TensorShape get_output_shape(ElementScalarUnaryAttrs const &, TensorShape const &) {
  NOT_IMPLEMENTED();
}

} // namespace FlexFlow
