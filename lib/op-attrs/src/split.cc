#include "op-attrs/ops/split.h"
#include "op-attrs/ff_dim.h"
#include "utils/containers.h"
#include "utils/exception.h"

namespace FlexFlow {

std::vector<ParallelTensorShape>
    get_output_shapes(SplitAttrs const &attrs,
                      ParallelTensorShape const &input) {

  std::size_t dims_sum = sum(attrs.splits);
  if (dims_sum != input.at(ff_dim_t(attrs.axis)).size) {
    throw mk_runtime_error(
        "SplitAttrs: dims_sum != input.at(ff_dim_t(attrs.axis)).size");
  }

  std::vector<ParallelTensorShape> outputs;
  for (std::size_t i = 0; i < attrs.splits.size(); ++i) {
    outputs.emplace_back(input);
    outputs.back().at(ff_dim_t(attrs.axis)).size = attrs.splits[i];
    outputs.back().at(ff_dim_t(attrs.axis)).degree =
        input.at(ff_dim_t(attrs.axis)).degree;
    outputs.back().at(ff_dim_t(attrs.axis)).is_replica_dim = attrs.axis == 0;
  }
  return outputs;
}

std::vector<ParallelTensorShape>
    get_output_shape(SplitAttrs const &attrs,
                     ParallelTensorShape const &input) {
  std::size_t dims_sum = sum(attrs.splits);
  if (dims_sum != input.at(ff_dim_t(attrs.axis)).size) {
    throw mk_runtime_error(
        "SplitAttrs: dims_sum != input.at(ff_dim_t(attrs.axis)).size");
  }

  std::vector<ParallelTensorShape> outputs;
  for (std::size_t i = 0; i < attrs.splits.size(); ++i) {
  }
  return outputs;
}

} // namespace FlexFlow
