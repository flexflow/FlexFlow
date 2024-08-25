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

template <typename DT>
void compute_qkv(IncMultiHeadSelfAttentionMeta const *m,
                 BatchConfig const *bc,
                 int shard_id,
                 DT const *input_ptr,
                 DT const *weight_ptr,
                 DT *output_ptr,
                 DT const *bias_ptr,
                 ffStream_t stream);

template <typename DT>
void apply_pos_encoding(IncMultiHeadSelfAttentionMeta const *m,
                        BatchConfig const *bc,
                        DT *output_ptr,
                        cudaStream_t stream);

template <typename DT>
void update_qkv_cache(IncMultiHeadSelfAttentionMeta const *m,
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
