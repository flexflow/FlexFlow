#include "op-attrs/ops/layer_norm.h"

namespace FlexFlow {

bool LayerNormAttrs::is_valid(ParallelTensorShape const &input) const {
  if (!input.is_valid()) {
    return false;
  }
  if (input.num_dims() < 2) {
    return false;
  }
  return true;
}

// todo: maybe we need to set the degree of parallel_dim
ParallelTensorShape get_output_shape(LayerNormAttrs const &attrs,
                                     ParallelTensorShape const &input) {
  ParallelTensorShape output = input;
  return output;
}

} // namespace FlexFlow
