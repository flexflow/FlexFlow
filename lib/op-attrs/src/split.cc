#include "op-attrs/ops/split.h"
#include "op-attrs/ff_dim.h"

namespace FlexFlow {

bool SplitAttrs::is_valid(ParallelTensorShape const &input) const {
  if (!input.is_valid()) {
    return false;
  }
  std::size_t dims_sum = 0;

  for (std::size_t i = 0; i < this->splits.size(); ++i) {
    dims_sum += splits[i];
  }

  if (dims_sum != input.at(ff_dim_t(axis)).size) {
    return false;
  }
  return true;
}

std::vector<ParallelTensorShape>
    get_output_shapes(SplitAttrs const &attrs,
                      ParallelTensorShape const &input) {

  assert(attrs.is_valid(input));
  std::vector<ParallelTensorShape> outputs;
  for (std::size_t i = 0; i < attrs.splits.size(); ++i) {
    outputs.emplace_back(input);
    outputs.back().at(ff_dim_t(attrs.axis)).size = attrs.splits[i];
  }
  return outputs;
}

} // namespace FlexFlow
