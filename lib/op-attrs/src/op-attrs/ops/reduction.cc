#include "op-attrs/ops/reduction.h"

namespace FlexFlow {

ParallelTensorShape get_output_shape(ReductionAttrs const &attrs, ParallelTensorShape const &input_shape) { 
  NOT_IMPLEMENTED();
}

/* ParallelTensorShape ReductionAttrs::output_shape(ParallelTensorShape const
 * &input_shape) const { */
/*   ParallelTensorShape output = input_shape; */
/*   output.at(this->reduction_legion_dim).degree /= this->reduction_degree; */
/*   output.at(this->reduction_legion_dim).size /= this->reduction_degree; */
/*   return output; */
/* } */

} // namespace FlexFlow
