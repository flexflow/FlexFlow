#include "op-attrs/ops/reduction.h"

namespace FlexFlow {

/* ParallelTensorShape ReductionAttrs::output_shape(ParallelTensorShape const
 * &input_shape) const { */
/*   ParallelTensorShape output = input_shape; */
/*   output.at(this->reduction_legion_dim).degree /= this->reduction_degree; */
/*   output.at(this->reduction_legion_dim).size /= this->reduction_degree; */
/*   return output; */
/* } */

ParallelTensorShape get_output_shape(ReductionAttrs const &attrs,
                                     ParallelTensorShape const &input_shape) {
  ParallelTensorShape output(input_shape.dims, input_shape.data_type);
  output.at(attrs.reduction_dim).size = 1;
  return output;
}

} // namespace FlexFlow
