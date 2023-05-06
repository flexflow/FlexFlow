#ifndef _FLEXFLOW_INC_MULTIHEAD_SELF_ATTENTION_VERIFY_PARAMS_H
#define _FLEXFLOW_INC_MULTIHEAD_SELF_ATTENTION_VERIFY_PARAMS_H

#include "flexflow/fftype.h"
#include "flexflow/parallel_tensor.h"

namespace FlexFlow {

struct IncMultiHeadSelfAttentionVerifyParams {
  LayerID layer_guid;
  int embed_dim, num_heads, kdim, vdim;
  float dropout;
  bool bias, add_bias_kv, add_zero_attn, apply_rotary_embedding;

  bool is_valid(ParallelTensorShape const &) const;
};

bool operator==(IncMultiHeadSelfAttentionVerifyParams const &,
                IncMultiHeadSelfAttentionVerifyParams const &);

} // namespace FlexFlow

namespace std {
template <>
struct hash<FlexFlow::IncMultiHeadSelfAttentionVerifyParams> {
  size_t
      operator()(FlexFlow::IncMultiHeadSelfAttentionVerifyParams const &) const;
};
} // namespace std

#endif // _FLEXFLOW_INC_MULTIHEAD_SELF_ATTENTION_VERIFY_PARAMS_H
