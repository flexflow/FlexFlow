#ifndef _FLEXFLOW_SPECINFER_INC_MULTIHEAD_SELF_ATTENTION_PARAMS_H
#define _FLEXFLOW_SPECINFER_INC_MULTIHEAD_SELF_ATTENTION_PARAMS_H

#include "flexflow/ffconst.h"
#include "flexflow/fftype.h"
#include "flexflow/parallel_tensor.h"

namespace FlexFlow {

struct SpecInferIncMultiHeadSelfAttentionParams {
  LayerID layer_guid;
  int embed_dim, num_q_heads, num_kv_heads, kdim, vdim;
  float dropout, scaling_factor;
  bool qkv_bias, final_bias, add_zero_attn, apply_rotary_embedding,
      scaling_query, qk_prod_scaling, position_bias;

  bool is_valid(ParallelTensorShape const &) const;
};

bool operator==(SpecInferIncMultiHeadSelfAttentionParams const &,
                SpecInferIncMultiHeadSelfAttentionParams const &);

} // namespace FlexFlow

namespace std {
template <>
struct hash<FlexFlow::SpecInferIncMultiHeadSelfAttentionParams> {
  size_t
      operator()(FlexFlow::SpecInferIncMultiHeadSelfAttentionParams const &) const;
};
} // namespace std

#endif // _FLEXFLOW_SPECINFER_INC_MULTIHEAD_SELF_ATTENTION_PARAMS_H
