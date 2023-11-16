#ifndef _FLEXFLOW_PCG_INCLUDE_PCG_FILE_FORMAT_V1_OPS_ATTENTION_ATTRS_H
#define _FLEXFLOW_PCG_INCLUDE_PCG_FILE_FORMAT_V1_OPS_ATTENTION_ATTRS_H

#include "op-attrs/ops/attention.h"
#include "utils/visitable.h"

namespace FlexFlow {

struct V1MultiHeadAttentionAttrs {
  req<int> embed_dim, num_heads, kdim, vdim;
  req<float> dropout;
  req<bool> bias, add_bias_kv, add_zero_attn;
};
FF_VISITABLE_STRUCT(V1MultiHeadAttentionAttrs,
                    embed_dim,
                    num_heads,
                    kdim,
                    vdim,
                    dropout,
                    bias,
                    add_bias_kv,
                    add_zero_attn);
CHECK_IS_JSONABLE(V1MultiHeadAttentionAttrs);

V1MultiHeadAttentionAttrs to_v1(MultiHeadAttentionAttrs const &a);
MultiHeadAttentionAttrs from_v1(V1MultiHeadAttentionAttrs const &va);

} // namespace FlexFlow

#endif
