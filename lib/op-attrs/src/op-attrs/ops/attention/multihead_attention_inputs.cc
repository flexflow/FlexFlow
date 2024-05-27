#include "op-attrs/ops/attention/multihead_attention_inputs.h"
#include "op-attrs/tensor_shape.h"

namespace FlexFlow {

template <typename T>
static bool all_same(T const &x, T const &y, T const &z) {
  return x == y && y == z;
}

tl::expected<MultiHeadAttentionInputs, std::string> parse_attention_input_shape(TensorShape const &input_q,
                                                     TensorShape const &input_k,
                                                     TensorShape const &input_v) {
  if (num_dims(input_q) != 3) {
    return tl::unexpected(fmt::format("Query input has incorrect number of dims: {} != {}", num_dims(input_q), 3));
  }
  if (num_dims(input_k) != 3) {
    return tl::unexpected(fmt::format("Key input has incorrect number of dims: {} != {}", num_dims(input_k), 3));
  }
  if (num_dims(input_v) != 3) {
    return tl::unexpected(fmt::format("Value input has incorrect number of dims: {} != {}", num_dims(input_v), 3));
  }

  size_t seq_len_q = dim_at_idx(input_q, ff_dim_t{-2});
  size_t seq_len_k = dim_at_idx(input_k, ff_dim_t{-2});
  size_t seq_len_v = dim_at_idx(input_v, ff_dim_t{-2});

  if (!all_same(seq_len_q, seq_len_k, seq_len_v)) {
    return tl::unexpected(fmt::format("Q, K, V disagree on the sequence length: {} (Q) vs {} (K) vs {} (V)", seq_len_q, seq_len_k, seq_len_v));
  }

  size_t batch_size_q = dim_at_idx(input_q, ff_dim_t{-3});
  size_t batch_size_k = dim_at_idx(input_k, ff_dim_t{-3});
  size_t batch_size_v = dim_at_idx(input_v, ff_dim_t{-3});

  if (!all_same(batch_size_q, batch_size_k, batch_size_v)) {
    return tl::unexpected(fmt::format("Q, K, V disagree on the batch size: {} (Q) vs {} (K) vs {} (V)", batch_size_q, batch_size_k, batch_size_v));
  }

  if (!all_same(input_q.data_type, input_k.data_type, input_v.data_type)) {
    return tl::unexpected(fmt::format("Q, K, V disagree on the datatype: {} (Q) vs {} (K) vs {} (V)", input_q.data_type, input_k.data_type, input_v.data_type));
  }

  size_t q_size = dim_at_idx(input_q, ff_dim_t{-1});
  size_t k_size = dim_at_idx(input_k, ff_dim_t{-1});
  size_t v_size = dim_at_idx(input_v, ff_dim_t{-1});

  return MultiHeadAttentionInputs{
    batch_size_q,
    seq_len_q, 
    q_size,
    k_size,
    v_size,
    input_q.data_type,
  };
}

} // namespace FlexFlow
