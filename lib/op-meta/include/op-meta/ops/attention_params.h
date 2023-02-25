#ifndef _FLEXFLOW_ATTENTION_PARAMS_H
#define _FLEXFLOW_ATTENTION_PARAMS_H

#include "op-meta/parallel_tensor_shape.h"
#include "op-meta/ops/op_params.h"
#include "visit_struct/visit_struct.hpp"

namespace FlexFlow {
namespace opmeta {

struct MultiHeadAttentionParams : public OpParamsInterface {
public:
  int num_outputs(std::vector<ParallelTensorShape> const &inputs) const override;
  bool is_valid(std::vector<ParallelTensorShape> const &inputs) const override;
  std::vector<ParallelTensorShape> output_shapes(std::vector<ParallelTensorShape> const &inputs) const override;
  OperatorType op_type() const override;
public:
  int embed_dim, num_heads, kdim, vdim;
  float dropout;
  bool bias, add_bias_kv, add_zero_attn;

};

bool operator==(MultiHeadAttentionParams const &, MultiHeadAttentionParams const &);
bool operator<(MultiHeadAttentionParams const &, MultiHeadAttentionParams const &);
}
}

VISITABLE_STRUCT(::FlexFlow::opmeta::MultiHeadAttentionParams, 
                 embed_dim, num_heads, kdim, vdim, dropout, bias, add_bias_kv, add_zero_attn);

namespace std {
template <>
struct hash<::FlexFlow::opmeta::MultiHeadAttentionParams> {
  size_t operator()(::FlexFlow::opmeta::MultiHeadAttentionParams const &) const;
};
}

#endif 
