#include "op-attrs/ops/cast.h"

namespace FlexFlow {

ParallelTensorShape get_output_shape(CastAttrs const &attrs,
                                     ParallelTensorShape const &input) {
  ParallelTensorShape output = input;
  output.data_type = attrs.dtype;
}

/* bool CastAttrs::is_valid(ParallelTensorShape const &input) const { */
/*   bool valid = input.is_valid(); */
/*   valid &= (input.at(input.num_dims() - 1).degree == 1); */
/*   return valid; */
/* } */

} // namespace FlexFlow
