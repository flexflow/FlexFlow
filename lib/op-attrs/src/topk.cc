#include "op-attrs/ops/topk.h"
#include "utils/exception.h"

namespace FlexFlow {

ParallelTensorShape get_output_shape(TopKAttrs const &attrs,
                                     ParallelTensorShape const &input) {

  if (attrs.k > input.at(ff_dim_t(attrs.axis)).size) {
    throw mk_runtime_error(
        "TopKAttrs: k > input.at(ff_dim_t(attrs.axis)).size");
  }

  ParallelTensorShape output = input;
  output.at(ff_dim_t(attrs.axis)).size = attrs.k;
  output.at(ff_dim_t(attrs.axis)).degree =
      input.at(ff_dim_t(attrs.axis)).degree;
  output.at(ff_dim_t(attrs.axis)).is_replica_dim =
      input.at(ff_dim_t(attrs.axis)).degree > 1;
  return output;
}

} // namespace FlexFlow
