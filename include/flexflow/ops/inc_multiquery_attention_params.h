#ifndef _FLEXFLOW_INC_MULTIQUERY_ATTENTION_PARAMS_H
#define _FLEXFLOW_INC_MULTIQUERY_ATTENTION_PARAMS_H

#include "flexflow/fftype.h"
#include "flexflow/parallel_tensor.h"

namespace FlexFlow {

struct IncMultiQueryAttentionParams {
  LayerID layer_guid;
  int embed_dim, num_heads, kdim, vdim;
  float dropout;
  bool bias, add_bias_kv, add_zero_attn;

  bool is_valid(ParallelTensorShape const &) const;
};

bool operator==(IncMultiQueryAttentionParams const &,
                IncMultiQueryAttentionParams const &);

} // namespace FlexFlow

namespace std {
template <>
struct hash<FlexFlow::IncMultiQueryAttentionParams> {
  size_t operator()(FlexFlow::IncMultiQueryAttentionParams const &) const;
};
} // namespace std

#endif // _FLEXFLOW_INC_MULTIQUERY_ATTENTION_PARAMS_H
