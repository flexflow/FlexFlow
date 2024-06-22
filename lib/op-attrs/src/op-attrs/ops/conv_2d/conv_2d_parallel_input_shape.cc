#include "op-attrs/ops/conv_2d/conv_2d_parallel_input_shape.h"
#include "op-attrs/parallel_tensor_shape.h"

namespace FlexFlow {

Conv2DParallelInputShape
    parse_parallel_input_shape(ParallelTensorShape const &input) {
  assert(num_shard_dims(input) == 4);

  ShardParallelDim sample_dim = shard_dim_at_idx(input, ff_dim_t{0});
  ShardParallelDim channel_dim = shard_dim_at_idx(input, ff_dim_t{1});
  ShardParallelDim height_dim = shard_dim_at_idx(input, ff_dim_t{2});
  ShardParallelDim width_dim = shard_dim_at_idx(input, ff_dim_t{3});

  Conv2DParallelInputShape parsed = Conv2DParallelInputShape{
      sample_dim,
      channel_dim,
      height_dim,
      width_dim,
      get_sum_degree(input),
      get_discard_copy_degree(input),
      input.data_type,
  };

  assert(parsed.height_dim.degree == 1);
  assert(parsed.width_dim.degree == 1);

  return parsed;
}

} // namespace FlexFlow
