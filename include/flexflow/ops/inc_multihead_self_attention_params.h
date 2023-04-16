#ifndef _FLEXFLOW_INC_MULTIHEAD_SELF_ATTENTION_PARAMS_H
#define _FLEXFLOW_INC_MULTIHEAD_SELF_ATTENTION_PARAMS_H

#include "flexflow/fftype.h"
#include "flexflow/parallel_tensor.h"

namespace FlexFlow {

struct IncMultiHeadSelfAttentionParams {
  LayerID layer_guid;
  int embed_dim, num_heads, kdim, vdim;
  float dropout;
  bool bias, add_bias_kv, add_zero_attn, apply_rotary_embedding;

  bool is_valid(ParallelTensorShape const &) const;
};

bool operator==(IncMultiHeadSelfAttentionParams const &,
                IncMultiHeadSelfAttentionParams const &);

} // namespace FlexFlow

namespace std {
template <>
struct hash<FlexFlow::IncMultiHeadSelfAttentionParams> {
  size_t operator()(FlexFlow::IncMultiHeadSelfAttentionParams const &) const;
};
} // namespace std

#endif // _FLEXFLOW_INC_MULTIHEAD_SELF_ATTENTION_PARAMS_H
