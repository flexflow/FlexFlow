#ifndef _FLEXFLOW_ATTENTION_ATTRS_H
#define _FLEXFLOW_ATTENTION_ATTRS_H

#include "core.h"
#include "op-attrs/parallel_tensor_shape.h"
#include "utils/visitable.h"

namespace FlexFlow {

struct MultiHeadAttentionAttrs : use_visitable_cmp<MultiHeadAttentionAttrs> {
public:
  MultiHeadAttentionAttrs() = delete;
  MultiHeadAttentionAttrs(int embed_dim,
                          int num_heads,
                          int kdim,
                          int vdim,
                          float dropout,
                          bool bias,
                          bool add_bias_kv,
                          bool add_zero_attn);

public:
  int embed_dim, num_heads, kdim, vdim;
  float dropout;
  bool bias, add_bias_kv, add_zero_attn;
};

template <typename TensorType>
struct MultiHeadAttentionInputs
    : public use_visitable_cmp<MultiHeadAttentionInputs<TensorType>> {
public:
  MultiHeadAttentionInputs() = delete;

  MultiHeadAttentionInputs(TensorType const &query,
                           TensorType const &key,
                           TensorType const &value)
      : query(query), key(key), value(value) {}

  template <typename SubTensorType>
  MultiHeadAttentionInputs(MultiHeadAttentionInputs<SubTensorType> const &sub)
      : query(sub.query), key(sub.key), value(sub.value) {}

public:
  TensorType query;
  TensorType key;
  TensorType value;
};

int get_qProjSize(MultiHeadAttentionAttrs const &);
int get_vProjSize(MultiHeadAttentionAttrs const &);
int get_kProjSize(MultiHeadAttentionAttrs const &);
int get_oProjSize(MultiHeadAttentionAttrs const &);

int get_qSize(MultiHeadAttentionInputs<ParallelTensorShape> const &);
int get_kSize(MultiHeadAttentionInputs<ParallelTensorShape> const &);
int get_vSize(MultiHeadAttentionInputs<ParallelTensorShape> const &);
int get_oSize(ParallelTensorShape const &);

int get_qoSeqLength(MultiHeadAttentionInputs<ParallelTensorShape> const &);
int get_kvSeqLength(MultiHeadAttentionInputs<ParallelTensorShape> const &);

int get_num_samples(MultiHeadAttentionInputs<ParallelTensorShape> const &);

TensorShape get_weights_shape(MultiHeadAttentionAttrs const &,
                              MultiHeadAttentionInputs<TensorShape> const &);
ParallelTensorShape
    get_weights_shape(MultiHeadAttentionAttrs const &,
                      MultiHeadAttentionInputs<ParallelTensorShape> const &);

ParallelTensorShape
    get_output_shape(MultiHeadAttentionAttrs const &,
                     MultiHeadAttentionInputs<ParallelTensorShape> const &);
TensorShape get_output_shape(MultiHeadAttentionAttrs const &,
                             MultiHeadAttentionInputs<TensorShape> const &);

} // namespace FlexFlow

VISITABLE_STRUCT(::FlexFlow::MultiHeadAttentionAttrs,
                 embed_dim,
                 num_heads,
                 kdim,
                 vdim,
                 dropout,
                 bias,
                 add_bias_kv,
                 add_zero_attn);
MAKE_VISIT_HASHABLE(::FlexFlow::MultiHeadAttentionAttrs);

namespace FlexFlow {
static_assert(is_valid_opattr<MultiHeadAttentionAttrs>::value,
              "MultiHeadAttentionAttrs must be a valid opattr (see core.h)");
}

#endif
