#ifndef _FLEXFLOW_ATTENTION_ATTRS_H
#define _FLEXFLOW_ATTENTION_ATTRS_H

#include "core.h"
#include "op-attrs/ops/attention_attrs.dtg.h"
#include "op-attrs/parallel_tensor_shape.dtg.h"
#include "op-attrs/ops/attention_inputs.dtg.h"
#include "op-attrs/ops/parallel_attention_inputs.dtg.h"

namespace FlexFlow {

int get_qProjSize(MultiHeadAttentionAttrs const &);
int get_vProjSize(MultiHeadAttentionAttrs const &);
int get_kProjSize(MultiHeadAttentionAttrs const &);
int get_oProjSize(MultiHeadAttentionAttrs const &);

int get_qSize(ParallelMultiHeadAttentionInputs const &);
int get_qSize(MultiHeadAttentionInputs const &);

int get_kSize(ParallelMultiHeadAttentionInputs const &);
int get_kSize(MultiHeadAttentionInputs const &);

int get_vSize(ParallelMultiHeadAttentionInputs const &);
int get_vSize(MultiHeadAttentionInputs const &);

int get_oSize(ParallelTensorShape const &);
int get_oSize(TensorShape const &);

int get_qoSeqLength(ParallelMultiHeadAttentionInputs const &);
int get_qoSeqLength(MultiHeadAttentionInputs const &);

int get_kvSeqLength(ParallelMultiHeadAttentionInputs const &);
int get_kvSeqLength(MultiHeadAttentionInputs const &);

int get_num_samples(ParallelMultiHeadAttentionInputs const &);
int get_num_samples(MultiHeadAttentionInputs const &);

TensorShape get_weights_shape(MultiHeadAttentionAttrs const &,
                              MultiHeadAttentionInputs const &);
ParallelTensorShape
    get_weights_shape(MultiHeadAttentionAttrs const &,
                      ParallelMultiHeadAttentionInputs const &);

TensorShape get_output_shape(MultiHeadAttentionAttrs const &,
                             MultiHeadAttentionInputs const &);
ParallelTensorShape
    get_output_shape(MultiHeadAttentionAttrs const &,
                     ParallelMultiHeadAttentionInputs const &);

CHECK_VALID_OP_ATTR(MultiHeadAttentionAttrs);
} // namespace FlexFlow

#endif
