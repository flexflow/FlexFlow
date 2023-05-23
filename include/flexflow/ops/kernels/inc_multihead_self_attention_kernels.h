#ifndef _FLEXFLOW_OPS_KERNELS_INC_MULTIHEAD_SELF_ATTENTION_KERNELS_H
#define _FLEXFLOW_OPS_KERNELS_INC_MULTIHEAD_SELF_ATTENTION_KERNELS_H

#include "flexflow/device.h"
#include "flexflow/fftype.h"
#include "flexflow/op_meta.h"

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

} // namespace IncMultiHeadAttention
} // namespace Kernels
} // namespace FlexFlow

#endif // _FLEXFLOW_OPS_KERNELS_INC_MULTIHEAD_SELF_ATTENTION_KERNELS_H
