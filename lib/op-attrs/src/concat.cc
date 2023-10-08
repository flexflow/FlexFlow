#include "op-attrs/ops/concat.h"

namespace FlexFlow {

bool ConcatAttrs::is_valid(
    std::vector<ParallelTensorShape> const &input) const {
  bool valid = true;
  for (auto p : input) {
    valid &= p.is_valid();
  }
  return valid;
}

ParallelTensorShape
    get_output_shape(ConcatAttrs const &attrs,
                     std::vector<ParallelTensorShape> const &inputs) {
  ParallelTensorShape output = inputs[0];
  for (auto &i : inputs) {
    output.at(attrs.axis).size += i.at(attrs.axis).size;
  }
}

} // namespace FlexFlow
