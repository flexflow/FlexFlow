#include "op-attrs/ops/embedding.h"

namespace FlexFlow {

TensorShape get_weights_shape(EmbeddingAttrs const &, TensorShape const &) {
  NOT_IMPLEMENTED();
}

ParallelTensorShape get_output_shape(EmbeddingAttrs const &, ParallelTensorShape const &) {
  NOT_IMPLEMENTED();
}

} // namespace FlexFlow
