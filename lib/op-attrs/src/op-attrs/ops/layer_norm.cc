#include "op-attrs/ops/layer_norm.h"

namespace FlexFlow {

ParallelTensorShape get_output_shape(LayerNormAttrs const &, ParallelTensorShape const &) { 
  NOT_IMPLEMENTED();
}

} // namespace FlexFlow
