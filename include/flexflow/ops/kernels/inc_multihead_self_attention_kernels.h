#ifndef _FLEXFLOW_OPS_KERNELS_INC_MULTIHEAD_SELF_ATTENTION_KERNELS_H
#define _FLEXFLOW_OPS_KERNELS_INC_MULTIHEAD_SELF_ATTENTION_KERNELS_H

#include "flexflow/device.h"
#include "flexflow/fftype.h"
#include "flexflow/op_meta.h"
#include "flexflow/batch_config.h"

namespace FlexFlow {
namespace Kernels {
namespace IncMultiHeadAttention {

__global__ void build_w_out_tensor(float const *weight_ptr,
                                   float *contiguous_weight_ptr,
                                   int vProjSize,
                                   int oProjSize,
                                   int num_heads,
                                   int qkv_weight_block_size);

__global__ void apply_proj_bias_w(float *input_ptr,
                                  float const *bias_ptr,
                                  int num_tokens,
                                  int oProjSize);

__global__ void apply_proj_bias_qkv(float *input_ptr,
                                    float const *bias_ptr,
                                    int num_tokens,
                                    int qProjSize,
                                    int kProjSize,
                                    int vProjSize,
                                    int num_heads,
                                    bool scaling_query,
                                    float scaling_factor);

__global__ void
    apply_rotary_embedding(float *input_ptr,
                           cuFloatComplex *complex_input,
                           BatchConfig::PerTokenInfo const *tokenInfos,
                           int qProjSize,
                           int kProjSize,
                           int num_heads,
                           int num_tokens,
                           int q_block_size,
                           int k_block_size,
                           int v_block_size,
                           bool q_tensor);
} // namespace IncMultiHeadAttention
} // namespace Kernels
} // namespace FlexFlow

#endif // _FLEXFLOW_OPS_KERNELS_INC_MULTIHEAD_SELF_ATTENTION_KERNELS_H
