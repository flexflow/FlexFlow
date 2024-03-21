#ifndef _FLEXFLOW_ATTENTION_PARAMS_H
#define _FLEXFLOW_ATTENTION_PARAMS_H

#include "flexflow/fftype.h"
#include "flexflow/parallel_tensor.h"

namespace FlexFlow {

struct MultiHeadAttentionParams {
  LayerID layer_guid;
  int embed_dim, num_heads, kdim, vdim;
  float dropout;
  bool bias, add_bias_kv, add_zero_attn;
  char name[MAX_OPNAME];

  bool is_valid(std::tuple<ParallelTensorShape,
                           ParallelTensorShape,
                           ParallelTensorShape> const &) const;
};

bool operator==(MultiHeadAttentionParams const &,
                MultiHeadAttentionParams const &);

} // namespace FlexFlow

namespace std {
template <>
struct hash<FlexFlow::MultiHeadAttentionParams> {
  size_t operator()(FlexFlow::MultiHeadAttentionParams const &) const;
};
} // namespace std

#endif // _FLEXFLOW_ATTENTION_PARAMS_H
