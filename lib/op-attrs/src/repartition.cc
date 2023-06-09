#include "op-attrs/ops/repartition.h"

namespace FlexFlow {

RepartitionAttrs::RepartitionAttrs(ff_dim_t _dim, int _degree)
    : repartition_dim(_dim), repartition_degree(_degree) {}

/* bool RepartitionAttrs::is_valid(ParallelTensorShape const &input_shape) const
 * { */
/*   ParallelDim dim = input_shape.at(this->repartition_legion_dim); */
/*   return (dim.size % this->repartition_degree * dim.degree == 0); */
/* } */

} // namespace FlexFlow
