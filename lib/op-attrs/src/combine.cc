#include "op-attrs/ops/combine.h"
#include "utils/hash-utils.h"

namespace FlexFlow {

ParallelTensorShape get_output_shape(CombineAttrs const &attrs,
                                     ParallelTensorShape const &input) {
  ParallelTensorShape output = input;
  output.at(attrs.combine_dim).degree /= attrs.combine_degree;
  return output;
}

} // namespace FlexFlow
