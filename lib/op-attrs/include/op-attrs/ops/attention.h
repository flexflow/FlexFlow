#ifndef _FLEXFLOW_ATTENTION_ATTRS_H
#define _FLEXFLOW_ATTENTION_ATTRS_H

#include "op-attrs/parallel_tensor_shape.h"
#include "utils/visitable_funcs.h"
#include "core.h"

namespace FlexFlow {

struct MultiHeadAttentionAttrs {
public:
  MultiHeadAttentionAttrs() = delete;
  MultiHeadAttentionAttrs(int embed_dim, int num_heads, int kdim, int vdim, float dropout, 
                          bool bias, bool add_bias_kv, bool add_zero_attn);
public:
  int embed_dim, num_heads, kdim, vdim;
  float dropout;
  bool bias, add_bias_kv, add_zero_attn;
};

bool operator==(MultiHeadAttentionAttrs const &, MultiHeadAttentionAttrs const &);
bool operator!=(MultiHeadAttentionAttrs const &, MultiHeadAttentionAttrs const &);
bool operator<(MultiHeadAttentionAttrs const &, MultiHeadAttentionAttrs const &);

int qProjSize(MultiHeadAttentionAttrs const &attrs);
int vProjSize(MultiHeadAttentionAttrs const &attrs);
int kProjSize(MultiHeadAttentionAttrs const &attrs);
int oProjSize(MultiHeadAttentionAttrs const &attrs);

}

VISITABLE_STRUCT(::FlexFlow::MultiHeadAttentionAttrs, 
                 embed_dim, num_heads, kdim, vdim, dropout, bias, add_bias_kv, add_zero_attn);

namespace std {
template <>
struct hash<::FlexFlow::MultiHeadAttentionAttrs> {
  size_t operator()(::FlexFlow::MultiHeadAttentionAttrs const &) const;
};
}

namespace FlexFlow {

static_assert(is_valid_opattr<MultiHeadAttentionAttrs>::value, "MultiHeadAttentionAttrs must be a valid opattr (see core.h)");

}

#endif 
