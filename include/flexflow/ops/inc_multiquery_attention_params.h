#ifndef _FLEXFLOW_INC_MULTIQUERY_ATTENTION_PARAMS_H
#define _FLEXFLOW_INC_MULTIQUERY_ATTENTION_PARAMS_H

#include "flexflow/fftype.h"
#include "flexflow/parallel_tensor.h"

namespace FlexFlow {

struct IncMultiQuerySelfAttentionParams {
  LayerID layer_guid;
  int embed_dim, num_heads, kdim, vdim;
  float dropout;
  bool bias, add_bias_kv, add_zero_attn;

  bool is_valid(ParallelTensorShape const &) const;
};

bool operator==(IncMultiQuerySelfAttentionParams const &,
                IncMultiQuerySelfAttentionParams const &);

} // namespace FlexFlow

namespace std {
template <>
struct hash<FlexFlow::IncMultiQuerySelfAttentionParams> {
  size_t operator()(FlexFlow::IncMultiQuerySelfAttentionParams const &) const;
};
} // namespace std

#endif // _FLEXFLOW_INC_MULTIQUERY_ATTENTION_PARAMS_H
