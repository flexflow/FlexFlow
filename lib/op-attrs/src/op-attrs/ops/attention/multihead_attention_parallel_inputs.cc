#include "op-attrs/ops/attention/multihead_attention_parallel_inputs.h"
#include "op-attrs/ops/attention/multihead_attention_inputs.h"
#include "op-attrs/parallel_tensor_shape.h"
#include "utils/containers/require_same.h"

namespace FlexFlow {

MultiHeadAttentionParallelInputs
    parse_attention_parallel_input_shape(ParallelTensorShape const &input_q,
                                         ParallelTensorShape const &input_k,
                                         ParallelTensorShape const &input_v) {
  num_ptensor_shard_dims_t input_num_shard_dims =
      require_same(num_shard_dims(input_q),
                   num_shard_dims(input_k),
                   num_shard_dims(input_v));

  ASSERT(input_num_shard_dims.value == 3);

  relative_ff_dim_t seq_dim_idx = relative_ff_dim_t{-2};

  ShardParallelDim seq_len_dim =
      require_same(shard_dim_at_idx(input_q, seq_dim_idx),
                   shard_dim_at_idx(input_k, seq_dim_idx),
                   shard_dim_at_idx(input_v, seq_dim_idx));

  ASSERT(seq_len_dim.degree == 1);

  relative_ff_dim_t batch_dim_idx = relative_ff_dim_t{-3};

  ShardParallelDim batch_dim =
      require_same(shard_dim_at_idx(input_q, batch_dim_idx),
                   shard_dim_at_idx(input_k, batch_dim_idx),
                   shard_dim_at_idx(input_v, batch_dim_idx));

  ShardParallelDim query_dim = shard_dim_at_idx(input_q, relative_ff_dim_t{-1});
  ASSERT(query_dim.degree == 1,
         fmt::format("Expected query tensor to have query dim parallel degree "
                     "1, but received degree {}",
                     query_dim.degree));

  ShardParallelDim key_dim = shard_dim_at_idx(input_k, relative_ff_dim_t{-1});
  ASSERT(key_dim.degree == 1,
         fmt::format("Expected key tensor to have key dim parallel degree 1, "
                     "but received degree {}",
                     key_dim.degree));

  ShardParallelDim value_dim = shard_dim_at_idx(input_v, relative_ff_dim_t{-1});
  ASSERT(value_dim.degree == 1,
         fmt::format("Expected value tensor to have value dim parallel degree "
                     "1, but received degree {}",
                     value_dim.degree));

  DiscardCopyDegree discard_copy_degree =
      require_same(input_q.dims.replica_dims.discard_copy_degree,
                   input_k.dims.replica_dims.discard_copy_degree,
                   input_v.dims.replica_dims.discard_copy_degree);

  DataType data_type =
      require_same(input_q.data_type, input_k.data_type, input_v.data_type);

  return MultiHeadAttentionParallelInputs{
      /*batch_dim=*/batch_dim,
      /*sequence_dim=*/seq_len_dim,
      /*query_dim=*/query_dim,
      /*key_dim=*/key_dim,
      /*value_dim=*/value_dim,
      /*discard_copy_degree=*/discard_copy_degree,
      /*datatype=*/data_type,
  };
}

} // namespace FlexFlow
