#include "op-attrs/ops/attention/multihead_attention_parallel_inputs.h"
#include "op-attrs/parallel_tensor_shape.h"
#include "op-attrs/ops/attention/multihead_attention_inputs.h"

namespace FlexFlow {

template <typename T>
static bool all_same(T const &x, T const &y, T const &z) {
  return x == y && y == z;
}

tl::expected<MultiHeadAttentionParallelInputs, std::string> parse_attention_parallel_input_shape(ParallelTensorShape const &input_q,
                                                                                         ParallelTensorShape const &input_k,
                                                                                         ParallelTensorShape const &input_v) {
  tl::expected<MultiHeadAttentionInputs, std::string> unpar_parse_result = parse_attention_input_shape(
      get_reduced_shape(input_q), get_reduced_shape(input_k), get_reduced_shape(input_v));
  if (!unpar_parse_result.has_value()) {
    return tl::unexpected(fmt::format("MHA unparallel input parsing failed with message: \"{}\"", unpar_parse_result.error()));
  }

  if (num_shard_dims(input_q) != 3) {
    return tl::unexpected(fmt::format("Query input has incorrect number of dims: {} != {}", num_shard_dims(input_q), 3));
  }
  if (num_shard_dims(input_k) != 3) {
    return tl::unexpected(fmt::format("Key input has incorrect number of dims: {} != {}", num_shard_dims(input_k), 3));
  }
  if (num_shard_dims(input_v) != 3) {
    return tl::unexpected(fmt::format("Value input has incorrect number of dims: {} != {}", num_shard_dims(input_v), 3));
  }

  ShardParallelDim seq_len_q = shard_dim_at_idx(input_q, ff_dim_t{-2});
  if (seq_len_q.degree != 1) {
    return tl::unexpected(fmt::format("Query sequence length parallel degree expected to be 1, but received degree {}", seq_len_q.degree));
  }

  ShardParallelDim seq_len_k = shard_dim_at_idx(input_k, ff_dim_t{-2});
  if (seq_len_k.degree != 1) {
    return tl::unexpected(fmt::format("Key sequence length parallel degree expected to be 1, but received degree {}", seq_len_k.degree));
  }

  ShardParallelDim seq_len_v = shard_dim_at_idx(input_v, ff_dim_t{-2});
  if (seq_len_v.degree != 1) {
    return tl::unexpected(fmt::format("Value sequence length parallel degree expected to be 1, but received degree {}", seq_len_v.degree));
  }

  ShardParallelDim batch_size_q = shard_dim_at_idx(input_q, ff_dim_t{-3});
  ShardParallelDim batch_size_k = shard_dim_at_idx(input_k, ff_dim_t{-3});
  ShardParallelDim batch_size_v = shard_dim_at_idx(input_v, ff_dim_t{-3});

  if (!all_same(batch_size_q.degree, batch_size_k.degree, batch_size_v.degree)) {
    return tl::unexpected(fmt::format("Q, K, V disagree on the parallel degree of the batch dimension: {} (Q) vs {} (K) vs {} (V)", batch_size_q.degree, batch_size_k.degree, batch_size_v.degree));
  }

  ShardParallelDim query_dim = shard_dim_at_idx(input_q, ff_dim_t{-1});
  if (query_dim.degree > 1) {
    return tl::unexpected(fmt::format("Expected query tensor to have query dim parallel degree 1, but received degree {}", query_dim.degree));
  }

  ShardParallelDim key_dim = shard_dim_at_idx(input_k, ff_dim_t{-1});
  if (key_dim.degree > 1) {
    return tl::unexpected(fmt::format("Expected key tensor to have key dim parallel degree 1, but received degree {}", key_dim.degree));
  }

  ShardParallelDim value_dim = shard_dim_at_idx(input_v, ff_dim_t{-1});
  if (value_dim.degree > 1) {
    return tl::unexpected(fmt::format("Expected value tensor to have value dim parallel degree 1, but received degree {}", value_dim.degree));
  }

  int discard_copy_q = get_discard_copy_degree(input_q);
  int discard_copy_k = get_discard_copy_degree(input_k);
  int discard_copy_v = get_discard_copy_degree(input_v);

  if (!all_same(discard_copy_q, discard_copy_k, discard_copy_v)) {
    return tl::unexpected(fmt::format("Q, K, V disagree on the discard-copy degree: {} (Q) vs {} (K) vs {} (V)", discard_copy_q, discard_copy_k, discard_copy_v));
  }

  return MultiHeadAttentionParallelInputs{
    batch_size_q,
    seq_len_q,
    query_dim,
    key_dim,
    value_dim,
    discard_copy_q, 
    input_q.data_type,
  };
  
  // return;
}

} // namespace FlexFlow
