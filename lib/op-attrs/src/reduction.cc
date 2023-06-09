#include "op-attrs/ops/reduction.h"

namespace FlexFlow {

ReductionAttrs::ReductionAttrs(ff_dim_t _reduction_dim, int _reduction_degree)
    : reduction_dim(_reduction_dim), reduction_degree(_reduction_degree) {}

/* ParallelTensorShape ReductionAttrs::output_shape(ParallelTensorShape const
 * &input_shape) const { */
/*   ParallelTensorShape output = input_shape; */
/*   output.at(this->reduction_legion_dim).degree /= this->reduction_degree; */
/*   output.at(this->reduction_legion_dim).size /= this->reduction_degree; */
/*   return output; */
/* } */

} // namespace FlexFlow
