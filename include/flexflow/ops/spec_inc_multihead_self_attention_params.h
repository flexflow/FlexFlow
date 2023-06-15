#ifndef _FLEXFLOW_SPEC_INC_MULTIHEAD_SELF_ATTENTION_PARAMS_H
#define _FLEXFLOW_SPEC_INC_MULTIHEAD_SELF_ATTENTION_PARAMS_H

#include "flexflow/fftype.h"
#include "flexflow/parallel_tensor.h"

namespace FlexFlow {

struct SpecIncMultiHeadSelfAttentionParams {
  LayerID layer_guid;
  int embed_dim, num_heads, kdim, vdim;
  float dropout, scaling_factor;
  bool bias, add_bias_kv, add_zero_attn, apply_rotary_embedding, scaling_query,
      qk_prod_scaling;
  bool output_bias;

  bool is_valid(ParallelTensorShape const &) const;
};

bool operator==(SpecIncMultiHeadSelfAttentionParams const &,
                SpecIncMultiHeadSelfAttentionParams const &);

} // namespace FlexFlow

namespace std {
template <>
struct hash<FlexFlow::SpecIncMultiHeadSelfAttentionParams> {
  size_t
      operator()(FlexFlow::SpecIncMultiHeadSelfAttentionParams const &) const;
};
} // namespace std

#endif // _FLEXFLOW_SPEC_INC_MULTIHEAD_SELF_ATTENTION_PARAMS_H
