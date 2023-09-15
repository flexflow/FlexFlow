#include "op-attrs/ops/reduction.h"

namespace FlexFlow {

ParallelTensorShape get_output_shape(ReductionAttrs const &attrs,
                                     ParallelTensorShape const &input_shape) {
  ParallelTensorShape output(input_shape.dims, input_shape.data_type);
  output.at(attrs.reduction_dim).degree /= attrs.reduction_degree;
  output.at(attrs.reduction_dim).size /= attrs.reduction_degree;
  return output;
}

} // namespace FlexFlow
