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

template <typename DT>
void compute_attention_kernel_generation(IncMultiHeadSelfAttentionMeta const *m,
                                         BatchConfig const *bc,
                                         DT *output_ptr,
                                         ffStream_t stream);

template <typename DT>
void compute_o_prod_bias(IncMultiHeadSelfAttentionMeta const *m,
                         BatchConfig const *bc,
                         int shard_id,
                         DT *output_ptr,
                         DT const *weight_ptr,
                         DT const *bias_ptr,
                         int num_tokens,
                         ffStream_t stream);

template <typename DT>
__global__ void apply_position_bias_qkprd(DT *input_ptr,
                                          int num_tokens,
                                          int num_total_tokens,
                                          int num_heads,
                                          int global_num_q_heads,
                                          int shard_id);

template <typename DT>
__global__ void apply_proj_bias_w(DT *input_ptr,
                                  DT const *bias_ptr,
                                  int num_tokens,
                                  int qkv_weight_size,
                                  int oProjSize);

template <typename DT>
__global__ void apply_proj_bias_qkv(DT *input_ptr,
                                    DT const *bias_ptr,
                                    int shard_id,
                                    int num_tokens,
                                    int qProjSize,
                                    int kProjSize,
                                    int vProjSize,
                                    int num_heads,
                                    int num_kv_heads,
                                    bool scaling_query,
                                    float scaling_factor);

#if defined(FF_USE_CUDA) || defined(FF_USE_HIP_CUDA)
template <typename DT>
__global__ void
    apply_rotary_embedding(DT *input_ptr,
                           cuFloatComplex *complex_input,
                           BatchConfig::PerTokenInfo const *tokenInfos,
                           int qProjSize,
                           int kProjSize,
                           int num_heads,
                           int num_tokens,
                           int num_kv_heads,
                           int q_block_size,
                           int k_block_size,
                           int q_array_size,
                           bool q_tensor);
#elif defined(FF_USE_HIP_ROCM)
template <typename DT>
__global__ void
    apply_rotary_embedding(DT *input_ptr,
                           hipFloatComplex *complex_input,
                           BatchConfig::PerTokenInfo const *tokenInfos,
                           int qProjSize,
                           int kProjSize,
                           int num_heads,
                           int num_tokens,
                           int num_kv_heads,
                           int q_block_size,
                           int k_block_size,
                           int q_array_size,
                           bool q_tensor);
#endif

template <typename DT>
void compute_qkv_kernel(IncMultiHeadSelfAttentionMeta const *m,
                        BatchConfig const *bc,
                        int shard_id,
                        DT const *input_ptr,
                        DT const *weight_ptr,
                        DT *output_ptr,
                        DT const *bias_ptr,
                        ffStream_t stream);

template <typename DT>
void pre_build_weight_kernel(IncMultiHeadSelfAttentionMeta const *m,
                             GenericTensorAccessorR const weight,
                             DataType data_type,
                             ffStream_t stream);
} // namespace IncMultiHeadAttention
} // namespace Kernels
} // namespace FlexFlow

#endif // _FLEXFLOW_OPS_KERNELS_INC_MULTIHEAD_SELF_ATTENTION_KERNELS_H
