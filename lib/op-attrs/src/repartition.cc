#include "op-attrs/ops/repartition.h"
#include "op-attrs/parallel_dim.h"
#include "utils/exception.h"

namespace FlexFlow {

// this may be wrong partition by n multiplies degree by n and keeps shape the
// same
ParallelTensorShape get_output_shape(RepartitionAttrs const &attrs,
                                     ParallelTensorShape const &input) {
  ParallelDim dim = input.at(attrs.repartition_dim);
  if (dim.size % attrs.repartition_degree * dim.degree != 0) {
    throw mk_runtime_error("RepartitionAttrs: input.at(attrs.repartition_dim) "
                           "attrs.repartition_degree * dim.degree != 0");
  }
  ParallelTensorShape output(input.dims, input.data_type);
  output.at(attrs.repartition_dim).degree *= attrs.repartition_degree;
  return output;
}

} // namespace FlexFlow
