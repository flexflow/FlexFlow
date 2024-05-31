#include "op-attrs/ops/combine.h"
#include "op-attrs/parallel_tensor_shape.h"

namespace FlexFlow {

tl::expected<ParallelTensorShape, std::string> get_output_shape(CombineAttrs const &attrs, ParallelTensorShape const &input) {
  ShardParallelDim input_dim = ({
    std::optional<ShardParallelDim> result = try_get_shard_dim_at_idx(input, attrs.combine_dim);
    if (!result.has_value()) {
      return tl::unexpected(fmt::format("Failed to get shard dim at index {} in parallel tensor shape {}", attrs.combine_dim, input));
    }

    result.value();
  });

  if (input_dim.degree % attrs.combine_degree != 0) {
    return tl::unexpected(fmt::format("Combine received tensor containing parallel dim {} with degree {}, which is not divisible by combine degree {}", attrs.combine_dim, input_dim.degree, attrs.combine_degree));
  }

  ParallelTensorShape output = input;
  shard_dim_at_idx(output, attrs.combine_dim).degree /= attrs.combine_degree;

  return output;
}

} // namespace FlexFlow
