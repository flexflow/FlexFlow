#ifndef _FLEXFLOW_ATTENTION_PARAMS_H
#define _FLEXFLOW_ATTENTION_PARAMS_H

#include "op-meta/parallel_tensor_shape.h"

namespace FlexFlow {

struct MultiHeadAttentionParams {
public:
  bool is_valid(std::tuple<ParallelTensorShape,
                           ParallelTensorShape,
                           ParallelTensorShape> const &) const;

  using AsConstTuple = std::tuple<int, int, int, int, float, bool, bool, bool>;
  AsConstTuple as_tuple() const;
public:
  int embed_dim, num_heads, kdim, vdim;
  float dropout;
  bool bias, add_bias_kv, add_zero_attn;

};

bool operator==(MultiHeadAttentionParams const &, MultiHeadAttentionParams const &);
bool operator<(MultiHeadAttentionParams const &, MultiHeadAttentionParams const &);

} // namespace FlexFlow

namespace std {
template <>
struct hash<FlexFlow::MultiHeadAttentionParams> {
  size_t operator()(FlexFlow::MultiHeadAttentionParams const &) const;
};
} // namespace std

#endif // _FLEXFLOW_ATTENTION_PARAMS_H
