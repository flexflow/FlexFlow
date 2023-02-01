#ifndef _FLEXFLOW_ATTENTION_PARAMS_H
#define _FLEXFLOW_ATTENTION_PARAMS_H

#include "op-meta/parallel_tensor_shape.h"
#include "op-meta/ops/op_params.h"

namespace FlexFlow {

struct MultiHeadAttentionParams : public OpParamsInterface {
public:
  using AsConstTuple = std::tuple<int, int, int, int, float, bool, bool, bool>;
  AsConstTuple as_tuple() const;

  int num_outputs(std::vector<ParallelTensorShape> const &inputs) const override;
  bool is_valid(std::vector<ParallelTensorShape> const &inputs) const override;
  OperatorType op_type() const override;
public:
  int embed_dim, num_heads, kdim, vdim;
  float dropout;
  bool bias, add_bias_kv, add_zero_attn;

};

bool operator==(MultiHeadAttentionParams const &, MultiHeadAttentionParams const &);
bool operator<(MultiHeadAttentionParams const &, MultiHeadAttentionParams const &);

}

namespace std {
template <>
struct hash<FlexFlow::MultiHeadAttentionParams> {
  size_t operator()(FlexFlow::MultiHeadAttentionParams const &) const;
};
}

#endif 
