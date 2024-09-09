#include "op-attrs/ops/topk.h"

namespace FlexFlow {

TensorShape get_output_shape(TopKAttrs const &,
                             TensorShape const &) {
  NOT_IMPLEMENTED();
}

ParallelTensorShape get_output_shape(TopKAttrs const &attrs,
                                     ParallelTensorShape const &input_shape) {
  NOT_IMPLEMENTED();
}

} // namespace FlexFlow
