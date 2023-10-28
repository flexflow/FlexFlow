#include "op-attrs/ops/layer_norm.h"
#include "utils/exceptions.h"

namespace FlexFlow {

// todo: maybe we need to set the degree of parallel_dim
ParallelTensorShape get_output_shape(LayerNormAttrs const &attrs,
                                     ParallelTensorShape const &input_shape) {
  if (input.num_dims() < 2) {
    throw mk_runtime_error("LayerNorm: input must have at least 2 dimensions");
  }
  // output shape is smae as input
  return input_shape;
}

} // namespace FlexFlow
