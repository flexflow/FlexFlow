#include "op-attrs/ops/layer_norm.h"

namespace FlexFlow {

TensorShape get_output_shape(LayerNormAttrs const &,
                             TensorShape const &) {
  NOT_IMPLEMENTED();
}

ParallelTensorShape get_output_shape(LayerNormAttrs const &,
                                     ParallelTensorShape const &) {
  NOT_IMPLEMENTED();
}

} // namespace FlexFlow
