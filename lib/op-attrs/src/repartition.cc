#include "op-attrs/ops/repartition.h"
#include "op-attrs/parallel_dim.h"

namespace FlexFlow {

/* bool RepartitionAttrs::is_valid(ParallelTensorShape const &input_shape) const
 * { */
/*   ParallelDim dim = input_shape.at(this->repartition_legion_dim); */
/*   return (dim.size % this->repartition_degree * dim.degree == 0); */
/* } */

bool RepartitionAttrs::is_valid(ParallelTensorShape const &input) const {
  if (!input.is_valid()) {
    return false;
  }
  ParallelDim dim = input.at(this->repartition_dim);
  return (dim.size % this->repartition_degree * dim.degree == 0);
}

//this may be wrong partition by n multiplies degree by n and keeps shape the same
ParallelTensorShape get_output_shape(RepartitionAttrs const &attrs,
                                     ParallelTensorShape const &input_shape) {
  ParallelTensorShape output(input_shape.dims, input_shape.data_type);
  output.at(attrs.repartition_dim).degree *= attrs.repartition_degree;
  return output;
}


} // namespace FlexFlow
