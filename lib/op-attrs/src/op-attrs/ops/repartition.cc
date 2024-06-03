#include "op-attrs/ops/repartition.h"

namespace FlexFlow {

tl::expected<ParallelTensorShape, std::string>
    get_output_shape(RepartitionAttrs const &attrs,
                     ParallelTensorShape const &input_shape) {
  ParallelTensorShape output_shape = input_shape;
  output_shape.dims.shard_dims.at(attrs.repartition_dim).degree *=
      attrs.repartition_degree;
  return output_shape;
}

} // namespace FlexFlow
