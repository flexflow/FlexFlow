#include "op-attrs/ops/combine.h"
#include "utils/hash-utils.h"

namespace FlexFlow {

bool CombineAttrs::is_valid(ParallelTensorShape const &input) const {
    if (!input.is_valid()) {
        return false;
    }
    return true;
}

ParallelTensorShape get_output_shape(CombineAttrs const & attrs,
                                     ParallelTensorShape const & input) {
  ParallelTensorShape output = input_shape;
  output.at(attrs.combine_dim).degree /= attrs.combine_degree;
  return output;                                     
}

} // namespace FlexFlow
