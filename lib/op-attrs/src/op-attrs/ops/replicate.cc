#include "op-attrs/ops/replicate.h"

namespace FlexFlow {

ParallelTensorShape get_output_shape(ReplicateAttrs const &attrs,
                                     ParallelTensorShape const &input_shape) {
  ParallelTensorShape output_shape = input_shape;
  output_shape.dims.replica_dims.discard_copy_degree.value *= attrs.replicate_degree;
  return output_shape;
}

} // namespace FlexFlow
