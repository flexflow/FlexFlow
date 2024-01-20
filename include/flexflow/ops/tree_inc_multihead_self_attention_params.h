#ifndef _FLEXFLOW_INC_MULTIHEAD_SELF_ATTENTION_VERIFY_PARAMS_H
#define _FLEXFLOW_INC_MULTIHEAD_SELF_ATTENTION_VERIFY_PARAMS_H

#include "flexflow/ffconst.h"
#include "flexflow/fftype.h"
#include "flexflow/parallel_tensor.h"

namespace FlexFlow {

struct TreeIncMultiHeadSelfAttentionParams {
  LayerID layer_guid;
  int embed_dim, num_q_heads, kdim, vdim, num_kv_heads,
      tensor_parallelism_degree;
  float dropout, scaling_factor;
  bool qkv_bias, final_bias, add_zero_attn, apply_rotary_embedding,
      scaling_query, qk_prod_scaling, position_bias;
  DataType quantization_type;
  bool offload;
  char name[MAX_OPNAME];
  bool is_valid(ParallelTensorShape const &) const;
};

bool operator==(TreeIncMultiHeadSelfAttentionParams const &,
                TreeIncMultiHeadSelfAttentionParams const &);

} // namespace FlexFlow

namespace std {
template <>
struct hash<FlexFlow::TreeIncMultiHeadSelfAttentionParams> {
  size_t
      operator()(FlexFlow::TreeIncMultiHeadSelfAttentionParams const &) const;
};
} // namespace std

#endif // _FLEXFLOW_INC_MULTIHEAD_SELF_ATTENTION_VERIFY_PARAMS_H
