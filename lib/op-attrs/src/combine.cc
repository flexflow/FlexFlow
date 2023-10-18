#include "op-attrs/ops/combine.h"
#include "utils/hash-utils.h"

namespace FlexFlow {
ParallelTensorShape
    get_output_shape_shape(CombineAttrs const &attrs,
                           ParallelTensorShape const &input_shape) {
  ParallelTensorShape output_shape = input_shape;
  output_shape.at(attrs.combine_dim).degree /= attrs.combine_degree;
  output_shape.at(attrs.combine_dim).is_replica_dim =
      output_shape.at(attrs.combine_dim).degree > 1;
  return output_shape;
}

} // namespace FlexFlow
