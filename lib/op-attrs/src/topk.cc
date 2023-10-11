#include "op-attrs/ops/topk.h"

namespace FlexFlow {

bool TopKAttrs::is_valid(ParallelTensorShape const &input) const {
  if (!input.is_valid()) {
    return false;
  }

  if (k > input.at(ff_dim_t(axis)).size) {
    return false;
  }
  return true;
}

ParallelTensorShape get_output_shape(TopKAttrs const &attrs,
                                     ParallelTensorShape const &input) {
  assert(attrs.is_valid(input));
  ParallelTensorShape output = input;
  output.at(ff_dim_t(attrs.axis)).size = attrs.k;
  return output;
}

} // namespace FlexFlow
