#include "op-attrs/ops/combine.h"
#include "utils/hash-utils.h"

namespace FlexFlow {

ParallelTensorShape output_shape(CombineAttrs const &attrs,
                                 ParallelTensorShape const &input_shape) {
  ParallelTensorShape output = input_shape;
  output.at(attrs.combine_dim).degree /= attrs.combine_degree;
  return output;
}

/* bool CombineAttrs::is_valid(ParallelTensorShape const &input) const { */
/*   return input.at(this->combine_legion_dim).degree % this->combine_degree ==
 * 0; */
/* } */

/* ParallelTensorShape CombineAttrs::output_shape(ParallelTensorShape const
 * &input_shape) const { */
/*   ParallelTensorShape output = input_shape; */
/*   output.at(this->combine_legion_dim).degree /= this->combine_degree; */
/*   return output; */
/* } */

} // namespace FlexFlow
