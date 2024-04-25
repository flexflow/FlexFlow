#include "op-attrs/ops/conv_2d/conv_2d_parallel_input_shape.h"
#include "op-attrs/parallel_tensor_shape.h"

namespace FlexFlow {

Conv2DParallelInputShape parse_parallel_input_shape(ParallelTensorShape const &input) {
  assert(num_shard_dims(input) == 4);
  
  ShardParallelDim sample_dim = shard_dim_at_idx(input, ff_dim_t{0});
  ShardParallelDim channel_dim = shard_dim_at_idx(input, ff_dim_t{1});
  ShardParallelDim height_dim = shard_dim_at_idx(input, ff_dim_t{2});
  ShardParallelDim width_dim = shard_dim_at_idx(input, ff_dim_t{3});

  return Conv2DParallelInputShape{
    sample_dim,
    channel_dim,
    height_dim,
    width_dim,
    input.dims.replica_dims.sum_degree,
    input.dims.replica_dims.discard_copy_degree,
    input.data_type,
  };
}


} // namespace FlexFlow
