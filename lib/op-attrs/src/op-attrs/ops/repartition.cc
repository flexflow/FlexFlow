#include "op-attrs/ops/repartition.h"

namespace FlexFlow {

ParallelTensorShape get_output_shape(RepartitionAttrs const &,
                                     ParallelTensorShape const &input_shape) {
  NOT_IMPLEMENTED();
}

/* bool RepartitionAttrs::is_valid(ParallelTensorShape const &input_shape) const
 * { */
/*   ParallelDim dim = input_shape.at(this->repartition_legion_dim); */
/*   return (dim.size % this->repartition_degree * dim.degree == 0); */
/* } */

} // namespace FlexFlow
