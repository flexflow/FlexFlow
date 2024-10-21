#ifndef _FLEXFLOW_OPS_KERNELS_INC_MULTIHEAD_SELF_ATTENTION_KERNELS_H
#define _FLEXFLOW_OPS_KERNELS_INC_MULTIHEAD_SELF_ATTENTION_KERNELS_H

#define QKV_WEIGHT_NUM 3
#define KV_WEIGHT_NUM 2

#include "flexflow/batch_config.h"
#include "flexflow/device.h"
#include "flexflow/fftype.h"
#include "flexflow/op_meta.h"
#include "flexflow/ops/inc_multihead_self_attention.h"

namespace FlexFlow {
namespace Kernels {
namespace IncMultiHeadAttention {

// kv layout: [num_pages, 2, page_size, num_kv_heads, head_dim]
__device__ __forceinline__ size_t get_k_entry_offset_verify(int const token_idx,
                                                     int const page_idx,
                                                     int const num_heads,
                                                     int const head_dim) {
  size_t index = ((page_idx) * kPagesize * 2 + (token_idx % kPagesize)) * head_dim * num_heads;
  return index;
}

// kv layout: [num_pages, 2, page_size, num_kv_heads, head_dim]
__device__ __forceinline__ size_t get_v_entry_offset_verify(int const token_idx,
                                                     int const page_idx,
                                                     int const num_heads,
                                                     int const head_dim) {
  size_t index = ((page_idx) * kPagesize * 2 + kPagesize + (token_idx % kPagesize)) * head_dim * num_heads;
  return index;
}

// // kv layout: [num_pages, 2, page_size, num_kv_heads, head_dim]
__device__ __forceinline__ size_t get_k_entry_offset(int const req_idx,
                                                     int const token_idx,
                                                     int const max_num_pages,
                                                     int const num_heads,
                                                     int const head_dim) {
  return ((req_idx * max_num_pages + token_idx / kPagesize) * kPagesize * 2 +
          token_idx % kPagesize) * /* page slot index */
         num_heads *
         head_dim;
}

// kv layout: [num_pages, 2, page_size, num_kv_heads, head_dim]
__device__ __forceinline__ size_t get_v_entry_offset(int const req_idx,
                                                     int const token_idx,
                                                     int const max_num_pages,
                                                     int const num_heads,
                                                     int const head_dim) {
  return ((req_idx * max_num_pages + token_idx / kPagesize) * kPagesize * 2 +
          kPagesize + token_idx % kPagesize) * /* page slot index */
         num_heads *
         head_dim;
}

template <typename DT>
void pre_build_weight(IncMultiHeadSelfAttentionMeta const *m,
                      GenericTensorAccessorR const weight,
                      DataType data_type,
                      ffStream_t stream);

// [For the tokens in batch]
// Compute qkv projection for the tokens in the batch.
template <typename DT>
void compute_qkv(IncMultiHeadSelfAttentionMeta const *m,
                 BatchConfig const *bc,
                 int shard_id,
                 DT const *input_ptr,
                 DT const *weight_ptr,
                 DT *output_ptr,
                 DT const *bias_ptr,
                 ffStream_t stream);

// [For the tokens in batch]
// Apply position embedding for qk.
// Note that this is only used for tokens in the current batch.
// For other Key tokens like in streaming cache, we nned other kernel to apply
// the position embedding.
template <typename DT>
void apply_pos_encoding_to_tokens_in_batch(
    IncMultiHeadSelfAttentionMeta const *m,
    BatchConfig const *bc,
    DT *output_ptr,
    cudaStream_t stream);

// [For the tokens in streaming cache]
// Apply position embedding for k projection in the streaming cache.
// Note that before the position encoding, the projection is moved *in order* to
// the kv memory took by the attention kernel. So our operation is applied where
// kvCache points to.
template <typename DT>
void apply_pos_encoding_to_streaming_proj(
    IncMultiHeadSelfAttentionMeta const *m,
    BatchConfig const *bc,
    cudaStream_t stream);

// [For the tokens in batch]
// Update the kv cache, and compact the q array.
// Source: qkv projeciton array of tokens in the batch.
// Destination: q&kv ptr took by the attention kernel.
// Note that the q&k here are the value after applying with position encoding.
template <typename DT>
void update_qkv_in_batch(IncMultiHeadSelfAttentionMeta const *m,
                         BatchConfig const *bc,
                         cudaStream_t stream);

void update_qkv_in_batch_verify(IncMultiHeadSelfAttentionMeta const *m,
                                BatchConfig const *bc,
                                cudaStream_t stream);

// [For the tokens in streaming cache]
// Convert the out-of-order cache to in-order relative position.
// Source: pre-pos-encoding kv values in the streaming cache.
// Destination: kv ptr took by the attention kernel.
template <typename DT>
void update_kv_in_streaming_cache(IncMultiHeadSelfAttentionMeta const *m,
                                  BatchConfig const *bc,
                                  cudaStream_t stream);

// [For the tokens in batch]
// Commit the kv values to the streaming cache.
// Source: qkv projeciton array of tokens in the batch.
// Destination: pre-pos-encoding kv values in the streaming cache.
template <typename DT>
void commit_kv(IncMultiHeadSelfAttentionMeta const *m,
               BatchConfig const *bc,
               cudaStream_t stream);

template <typename DT>
void produce_output(IncMultiHeadSelfAttentionMeta const *m,
                    BatchConfig const *bc,
                    DT *output_ptr,
                    cudaStream_t stream);

template <typename DT>
void compute_o_prod_bias(IncMultiHeadSelfAttentionMeta const *m,
                         BatchConfig const *bc,
                         int shard_id,
                         DT *output_ptr,
                         DT const *weight_ptr,
                         DT const *bias_ptr,
                         int num_tokens,
                         ffStream_t stream);
} // namespace IncMultiHeadAttention
} // namespace Kernels
} // namespace FlexFlow

#endif // _FLEXFLOW_OPS_KERNELS_INC_MULTIHEAD_SELF_ATTENTION_KERNELS_H
